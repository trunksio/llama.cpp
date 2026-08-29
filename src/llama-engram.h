#pragma once

// Detachable n-gram Engram adapter (AOA engrams, DeepSeek-style n-gram memory).
//
// An engram is a small trained lookup table keyed on hashed suffix n-grams of
// the token stream. At one decoder layer it adds a gated residual to the
// hidden states of a frozen model. Hash indices are computed on the host per
// ubatch (llm_graph_input_engram) from a per-sequence token history, so the
// memory works both for full-sequence scoring and incremental decoding.

#include "llama.h"
#include "llama-graph.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

struct llama_engram_config {
    uint32_t hidden_size    = 0;
    uint32_t heads_per_order = 0;
    uint32_t buckets        = 0;
    uint32_t head_dim       = 0;
    int32_t  injection_layer = 0;
    bool     use_gate       = true;
    float    residual_scale = 0.0f;

    std::vector<int32_t>  orders;
    std::vector<float>    gate_biases;   // one per order
    std::vector<int64_t>  hash_seeds;    // orders * heads_per_order, order-major
    std::set<int32_t>     ignored_tokens;

    uint32_t n_orders()      const { return (uint32_t) orders.size(); }
    uint32_t total_heads()   const { return n_orders() * heads_per_order; }
    uint32_t candidate_dim() const { return heads_per_order * head_dim; }
};

struct llama_engram {
    llama_engram_config config;

    // tensors, per order (outer) / head (inner)
    std::vector<std::vector<ggml_tensor *>> tables;     // [head_dim, buckets]
    std::vector<ggml_tensor *> w_hidden_key;            // [hidden, cd]
    std::vector<ggml_tensor *> w_memory_key;            // [cd, cd]
    std::vector<ggml_tensor *> w_value;                 // [cd, hidden]
    std::vector<ggml_tensor *> t_gate_bias;             // [1] per order

    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;

    // per-sequence token history indexed by position, filled during set_input;
    // mutable state so a const engram pointer in the graph params still works
    mutable std::map<llama_seq_id, std::vector<llama_token>> history;

    // fill idx (I32, [n_tokens * total_heads], head-slot-major) and valid
    // (F32, [n_tokens * n_orders], order-major) from the ubatch + history
    void compute_hashes(const llama_ubatch * ubatch, int32_t * idx, float * valid) const;
};

class llm_graph_input_engram : public llm_graph_input_i {
public:
    explicit llm_graph_input_engram(const llama_engram * engram) : engram(engram) {}
    virtual ~llm_graph_input_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * t_idx   = nullptr; // I32 [n_tokens * total_heads]
    ggml_tensor * t_valid = nullptr; // F32 [n_tokens * n_orders]

    const llama_engram * engram;
};

// build the gated residual on top of cur ([n_embd, n_tokens]) using the
// prepared input tensors; returns the enhanced hidden states
ggml_tensor * llama_engram_apply(
        ggml_context * ctx0,
        const llama_engram * engram,
        ggml_tensor * cur,
        ggml_tensor * t_idx,
        ggml_tensor * t_valid);

llama_engram * llama_engram_init_impl(const char * path);
