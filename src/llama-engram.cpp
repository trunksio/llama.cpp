#include "llama-engram.h"

#include "llama-impl.h"
#include "llama-batch.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

// hash constants — must match private_engram/hashing.py exactly
static const int64_t ENGRAM_TOKEN_MULT = 2862933555777941757LL;
static const int64_t ENGRAM_ORDER_MULT = 3935559000370003845LL;

static int64_t engram_wrap_mul(int64_t a, int64_t b) {
    return (int64_t) ((uint64_t) a * (uint64_t) b);
}

// torch.remainder semantics: non-negative result for a positive modulus
static int64_t engram_pos_mod(int64_t a, int64_t m) {
    int64_t r = a % m;
    return r < 0 ? r + m : r;
}

void llama_engram::compute_hashes(const llama_ubatch * ubatch, int32_t * idx, float * valid) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    const auto & cfg = config;

    const uint32_t n_orders = cfg.n_orders();
    const uint32_t heads    = cfg.heads_per_order;

    if (!ubatch->token) {
        // embedding input: no token ids, the engram stays silent
        std::fill(idx, idx + (size_t) n_tokens * cfg.total_heads(), 0);
        std::fill(valid, valid + (size_t) n_tokens * n_orders, 0.0f);
        return;
    }

    // update per-sequence history (position-indexed, robust to rewinds)
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq = ubatch->seq_id[i][0];
        const llama_pos    pos = ubatch->pos[i];
        auto & hist = history[seq];
        if ((size_t) pos >= hist.size()) {
            hist.resize(pos + 1, -1);
        }
        hist[pos] = ubatch->token[i];
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq = ubatch->seq_id[i][0];
        const llama_pos    pos = ubatch->pos[i];
        const auto & hist = history.at(seq);

        for (uint32_t oi = 0; oi < n_orders; ++oi) {
            const int32_t order = cfg.orders[oi];

            // gather the suffix window: component[offset] = token at pos-offset
            bool ok = true;
            std::vector<int64_t> comp(order);
            for (int32_t offset = 0; offset < order; ++offset) {
                const int64_t p = (int64_t) pos - offset;
                if (p < 0 || (size_t) p >= hist.size() || hist[p] < 0) {
                    ok = false;
                    break;
                }
                const llama_token tok = hist[p];
                if (cfg.ignored_tokens.count(tok) > 0) {
                    ok = false;
                    break;
                }
                comp[offset] = tok;
            }

            valid[(size_t) oi * n_tokens + i] = ok ? 1.0f : 0.0f;

            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t slot = oi * heads + h;
                int32_t index = 0;
                if (ok) {
                    int64_t state = cfg.hash_seeds[oi * heads + h];
                    for (int32_t offset = 0; offset < order; ++offset) {
                        const int64_t mixed = engram_wrap_mul(comp[offset] + 1 + offset, ENGRAM_TOKEN_MULT);
                        state ^= mixed;
                        state = engram_wrap_mul(state, ENGRAM_ORDER_MULT + (int64_t) (h + 1) * 2 + offset);
                        state ^= (state >> 29);
                    }
                    index = (int32_t) (1 + engram_pos_mod(state, cfg.buckets - 1));
                }
                idx[(size_t) slot * n_tokens + i] = index;
            }
        }
    }
}

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(t_idx && t_valid);

    std::vector<int32_t> idx((size_t) ubatch->n_tokens * engram->config.total_heads());
    std::vector<float>   valid((size_t) ubatch->n_tokens * engram->config.n_orders());

    engram->compute_hashes(ubatch, idx.data(), valid.data());

    ggml_backend_tensor_set(t_idx,   idx.data(),   0, idx.size()   * sizeof(int32_t));
    ggml_backend_tensor_set(t_valid, valid.data(), 0, valid.size() * sizeof(float));
}

bool llm_graph_input_engram::can_reuse(const llm_graph_params & params) {
    return params.engram == engram &&
           t_idx->ne[0] == (int64_t) params.ubatch.n_tokens * engram->config.total_heads();
}

ggml_tensor * llama_engram_apply(
        ggml_context * ctx0,
        const llama_engram * engram,
        ggml_tensor * cur,
        ggml_tensor * t_idx,
        ggml_tensor * t_valid,
        ggml_tensor ** out_norm2) {
    const auto & cfg = engram->config;

    const int64_t n_tokens = cur->ne[1];
    const float logit_scale = 1.0f / sqrtf((float) cfg.candidate_dim());

    ggml_tensor * cur_f32 = ggml_cast(ctx0, cur, GGML_TYPE_F32);
    ggml_tensor * total = nullptr;
    ggml_tensor * norms = nullptr; // [n_orders, n_tokens], squared per-order contribution norms

    for (uint32_t oi = 0; oi < cfg.n_orders(); ++oi) {
        ggml_tensor * candidate = nullptr;
        for (uint32_t h = 0; h < cfg.heads_per_order; ++h) {
            const uint32_t slot = oi * cfg.heads_per_order + h;
            ggml_tensor * slot_idx = ggml_view_1d(ctx0, t_idx, n_tokens,
                    (size_t) slot * n_tokens * sizeof(int32_t));
            ggml_tensor * rows = ggml_get_rows(ctx0, engram->tables[oi][h], slot_idx);
            candidate = candidate ? ggml_concat(ctx0, candidate, rows, 0) : rows;
        }

        ggml_tensor * valid = ggml_view_1d(ctx0, t_valid, n_tokens,
                (size_t) oi * n_tokens * sizeof(float));
        valid = ggml_reshape_2d(ctx0, valid, 1, n_tokens);

        ggml_tensor * gate;
        if (cfg.use_gate) {
            ggml_tensor * hk = ggml_mul_mat(ctx0, engram->w_hidden_key[oi], cur_f32);
            ggml_tensor * mk = ggml_mul_mat(ctx0, engram->w_memory_key[oi], candidate);
            ggml_tensor * logits = ggml_sum_rows(ctx0, ggml_mul(ctx0, hk, mk)); // [1, n_tokens]
            logits = ggml_scale(ctx0, logits, logit_scale);
            logits = ggml_add(ctx0, logits, engram->t_gate_bias[oi]);
            gate = ggml_sigmoid(ctx0, logits);
            gate = ggml_mul(ctx0, gate, valid);
        } else {
            gate = valid;
        }

        ggml_tensor * value   = ggml_mul_mat(ctx0, engram->w_value[oi], candidate); // [n_embd, n_tokens]
        ggml_tensor * contrib = ggml_mul(ctx0, value, gate);

        total = total ? ggml_add(ctx0, total, contrib) : contrib;

        if (out_norm2) {
            // squared norm of this order's contribution (at deployed scale)
            ggml_tensor * n2 = ggml_sum_rows(ctx0, ggml_sqr(ctx0, contrib)); // [1, n_tokens]
            n2 = ggml_scale(ctx0, n2, cfg.residual_scale * cfg.residual_scale);
            norms = norms ? ggml_concat(ctx0, norms, n2, 0) : n2;
        }
    }

    total = ggml_scale(ctx0, total, cfg.residual_scale);

    if (out_norm2) {
        // rows 0..n_orders-1: per-order squared norms; row n_orders: squared
        // norm of the full injected residual. The per-order rows carry the
        // provenance information: a stored key fires the high orders (their
        // windows contain the entity tokens) while a shared-furniture false
        // fire lights up only the lowest order.
        ggml_tensor * tot2 = ggml_sum_rows(ctx0, ggml_sqr(ctx0, total)); // [1, n_tokens]
        // [n_orders+1, n_tokens]: element (o, t) is order o's squared norm at
        // token t, with the full-residual squared norm as the final row
        *out_norm2 = ggml_cont(ctx0, ggml_concat(ctx0, norms, tot2, 0));
    }

    total = ggml_cast(ctx0, total, cur->type);

    return ggml_add(ctx0, cur, total);
}

//
// GGUF loading
//

static int32_t engram_kv_i32(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(std::string("engram: missing key ") + key);
    }
    return (int32_t) gguf_get_val_u32(ctx, id);
}

static float engram_kv_f32(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(std::string("engram: missing key ") + key);
    }
    return gguf_get_val_f32(ctx, id);
}

llama_engram * llama_engram_init_impl(const char * path) {
    ggml_context * ctx_meta = nullptr;

    gguf_init_params meta_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_meta,
    };

    gguf_context_ptr ctx_gguf { gguf_init_from_file(path, meta_params) };
    if (!ctx_gguf) {
        throw std::runtime_error(std::string("engram: failed to load ") + path);
    }
    ggml_context_ptr ctx_meta_ptr { ctx_meta };

    auto engram = std::make_unique<llama_engram>();
    auto & cfg = engram->config;

    cfg.hidden_size     = engram_kv_i32(ctx_gguf.get(), "engram.hidden_size");
    cfg.heads_per_order = engram_kv_i32(ctx_gguf.get(), "engram.heads_per_order");
    cfg.buckets         = engram_kv_i32(ctx_gguf.get(), "engram.buckets");
    cfg.head_dim        = engram_kv_i32(ctx_gguf.get(), "engram.head_dim");
    cfg.injection_layer = engram_kv_i32(ctx_gguf.get(), "engram.injection_layer");
    cfg.use_gate        = engram_kv_i32(ctx_gguf.get(), "engram.use_gate") != 0;
    cfg.residual_scale  = engram_kv_f32(ctx_gguf.get(), "engram.residual_scale");

    {
        const int64_t id = gguf_find_key(ctx_gguf.get(), "engram.orders");
        GGML_ASSERT(id >= 0 && gguf_get_arr_type(ctx_gguf.get(), id) == GGUF_TYPE_INT32);
        const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx_gguf.get(), id);
        cfg.orders.assign(data, data + gguf_get_arr_n(ctx_gguf.get(), id));
    }
    {
        const int64_t id = gguf_find_key(ctx_gguf.get(), "engram.gate_biases");
        GGML_ASSERT(id >= 0 && gguf_get_arr_type(ctx_gguf.get(), id) == GGUF_TYPE_FLOAT32);
        const float * data = (const float *) gguf_get_arr_data(ctx_gguf.get(), id);
        cfg.gate_biases.assign(data, data + gguf_get_arr_n(ctx_gguf.get(), id));
    }
    {
        const int64_t id = gguf_find_key(ctx_gguf.get(), "engram.hash_seeds");
        GGML_ASSERT(id >= 0 && gguf_get_arr_type(ctx_gguf.get(), id) == GGUF_TYPE_INT64);
        const int64_t * data = (const int64_t *) gguf_get_arr_data(ctx_gguf.get(), id);
        cfg.hash_seeds.assign(data, data + gguf_get_arr_n(ctx_gguf.get(), id));
    }
    {
        const int64_t id = gguf_find_key(ctx_gguf.get(), "engram.ignored_tokens");
        if (id >= 0) {
            const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx_gguf.get(), id);
            const size_t n = gguf_get_arr_n(ctx_gguf.get(), id);
            cfg.ignored_tokens = std::set<int32_t>(data, data + n);
        }
    }

    GGML_ASSERT(cfg.hash_seeds.size() == (size_t) cfg.total_heads());
    GGML_ASSERT(cfg.gate_biases.size() == cfg.orders.size());

    // build a ctx holding copies of all tensors (plus one bias tensor per order)
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf.get()) + cfg.n_orders();
    ggml_init_params ctx_params = {
        /* .mem_size   = */ (size_t) n_tensors * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    engram->ctx.reset(ggml_init(ctx_params));
    ggml_context * ctx = engram->ctx.get();

    auto dup_tensor = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * ref = ggml_get_tensor(ctx_meta_ptr.get(), name.c_str());
        if (!ref) {
            throw std::runtime_error("engram: missing tensor " + name);
        }
        ggml_tensor * t = ggml_dup_tensor(ctx, ref);
        ggml_set_name(t, name.c_str());
        return t;
    };

    engram->tables.resize(cfg.n_orders());
    for (uint32_t oi = 0; oi < cfg.n_orders(); ++oi) {
        const int32_t order = cfg.orders[oi];
        for (uint32_t h = 0; h < cfg.heads_per_order; ++h) {
            engram->tables[oi].push_back(dup_tensor("engram.table." + std::to_string(order) + "." + std::to_string(h)));
        }
        engram->w_hidden_key.push_back(dup_tensor("engram.hidden_key." + std::to_string(order)));
        engram->w_memory_key.push_back(dup_tensor("engram.memory_key." + std::to_string(order)));
        engram->w_value.push_back(dup_tensor("engram.value." + std::to_string(order)));

        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(bias, ("engram.gate_bias." + std::to_string(order)).c_str());
        engram->t_gate_bias.push_back(bias);
    }

    engram->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type()));
    if (!engram->buf) {
        throw std::runtime_error("engram: failed to allocate buffer");
    }

    // read tensor data from the file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("engram: cannot reopen ") + path);
    }
    const size_t data_offset = gguf_get_data_offset(ctx_gguf.get());
    std::vector<uint8_t> read_buf;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        const std::string name = ggml_get_name(t);
        if (name.rfind("engram.gate_bias.", 0) == 0) {
            continue; // filled from metadata below
        }
        const int64_t tid = gguf_find_tensor(ctx_gguf.get(), name.c_str());
        GGML_ASSERT(tid >= 0);
        const size_t offs = data_offset + gguf_get_tensor_offset(ctx_gguf.get(), tid);
        const size_t size = ggml_nbytes(t);
        read_buf.resize(size);
        file.seekg(offs);
        file.read((char *) read_buf.data(), size);
        if (!file) {
            throw std::runtime_error("engram: short read for tensor " + name);
        }
        ggml_backend_tensor_set(t, read_buf.data(), 0, size);
    }

    for (uint32_t oi = 0; oi < cfg.n_orders(); ++oi) {
        ggml_backend_tensor_set(engram->t_gate_bias[oi], &cfg.gate_biases[oi], 0, sizeof(float));
    }

    LLAMA_LOG_INFO("%s: loaded engram '%s': orders=%zu heads=%u buckets=%u head_dim=%u layer=%d scale=%.3f\n",
            __func__, path, cfg.orders.size(), cfg.heads_per_order, cfg.buckets, cfg.head_dim,
            cfg.injection_layer, cfg.residual_scale);

    return engram.release();
}

//
// public C API
//

llama_engram * llama_engram_init_from_file(const char * path) {
    try {
        return llama_engram_init_impl(path);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return nullptr;
    }
}

void llama_engram_free(llama_engram * engram) {
    delete engram;
}
