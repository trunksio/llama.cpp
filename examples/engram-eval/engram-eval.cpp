// Candidate-scoring evaluation for engram adapters.
//
// Reads a TSV of evaluation rows (prompt \t answer \t cand1|cand2|... \t prefix)
// and reports multiple-choice accuracy with and without a mounted engram,
// mirroring private_engram.evaluation.candidate_accuracy.

#include "llama.h"
#include "common.h"
#include "log.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        out.push_back(item);
    }
    return out;
}

struct eval_row {
    std::string prompt;
    std::string answer;
    std::vector<std::string> candidates;
    std::string prefix;
};

static double score_candidate(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::string & prompt,
        const std::string & continuation) {
    std::vector<llama_token> prompt_ids = common_tokenize(vocab, prompt, false, false);
    std::vector<llama_token> cont_ids   = common_tokenize(vocab, continuation, false, false);
    if (cont_ids.empty()) {
        return -1e30;
    }

    std::vector<llama_token> ids = prompt_ids;
    ids.insert(ids.end(), cont_ids.begin(), cont_ids.end());

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(batch, ids[i], (llama_pos) i, {0}, true);
    }

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("llama_decode failed\n");
        llama_batch_free(batch);
        return -1e30;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    double total = 0.0;
    // predicting token at position p+1 from logits at position p
    for (size_t p = prompt_ids.size() - 1; p + 1 < ids.size(); ++p) {
        const float * logits = llama_get_logits_ith(ctx, (int32_t) p);
        float maxl = -1e30f;
        for (int v = 0; v < n_vocab; ++v) {
            maxl = std::max(maxl, logits[v]);
        }
        double sum = 0.0;
        for (int v = 0; v < n_vocab; ++v) {
            sum += exp((double) logits[v] - maxl);
        }
        total += (double) logits[ids[p + 1]] - maxl - log(sum);
    }

    llama_batch_free(batch);
    return total / (double) cont_ids.size();
}

// greedy token-by-token generation: exercises the incremental-decode path,
// where the engram hashes each new token against its per-sequence history
static std::string generate_greedy(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::string & prompt,
        int n_predict,
        bool trace_signal = false) {
    llama_memory_clear(llama_get_memory(ctx), true);

    std::vector<llama_token> ids = common_tokenize(vocab, prompt, false, false);
    llama_batch batch = llama_batch_init((int32_t) std::max<size_t>(ids.size(), 1), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(batch, ids[i], (llama_pos) i, {0}, i + 1 == ids.size());
    }
    llama_engram_signal(ctx, nullptr, nullptr, nullptr, true);
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return "<decode failed>";
    }
    if (trace_signal) {
        float row[16]; int32_t stride = 0;
        if (llama_engram_signal_trace(ctx, row, 1, &stride) == 1) {
            fprintf(stderr, "signal last-prompt-pos:");
            for (int32_t o = 0; o < stride - 1; ++o) fprintf(stderr, " o%d=%.3f", o, row[o]);
            fprintf(stderr, " total=%.3f\n", row[stride - 1]);
        }
    }

    std::string out;
    llama_pos pos = (llama_pos) ids.size();
    int logits_idx = (int) ids.size() - 1;
    for (int step = 0; step < n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, logits_idx);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) best = v;
        }
        if (llama_vocab_is_eog(vocab, best)) break;
        out += common_token_to_piece(ctx, best);

        common_batch_clear(batch);
        common_batch_add(batch, best, pos++, {0}, true);
        if (llama_decode(ctx, batch) != 0) break;
        if (trace_signal) {
            float row[16]; int32_t stride = 0;
            if (llama_engram_signal_trace(ctx, row, 1, &stride) == 1) {
                std::string piece = common_token_to_piece(ctx, best);
                fprintf(stderr, "signal step %2d '%s':", step, piece.c_str());
                for (int32_t o = 0; o < stride - 1; ++o) fprintf(stderr, " o%d=%.3f", o, row[o]);
                fprintf(stderr, " total=%.3f\n", row[stride - 1]);
            }
        }
        logits_idx = 0;
    }
    llama_batch_free(batch);
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string engram_path;
    std::string lora_path;
    std::string eval_path;
    std::string gen_prompt;
    int n_predict = 24;
    int n_threads = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) { model_path = argv[++i]; }
        else if (arg == "--engram" && i + 1 < argc) { engram_path = argv[++i]; }
        else if (arg == "--lora" && i + 1 < argc) { lora_path = argv[++i]; }
        else if (arg == "--eval" && i + 1 < argc) { eval_path = argv[++i]; }
        else if (arg == "--generate" && i + 1 < argc) { gen_prompt = argv[++i]; }
        else if (arg == "--n-predict" && i + 1 < argc) { n_predict = atoi(argv[++i]); }
        else if (arg == "-t" && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else {
            fprintf(stderr, "usage: %s -m model.gguf (--eval rows.tsv | --generate PROMPT) "
                    "[--engram engram.gguf] [--n-predict N] [-t threads]\n", argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || (eval_path.empty() && gen_prompt.empty())) {
        fprintf(stderr, "model path and either --eval or --generate are required\n");
        return 1;
    }

    std::vector<eval_row> rows;
    if (!eval_path.empty()) {
        std::ifstream file(eval_path);
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto parts = split(line, '\t');
            if (parts.size() < 3) continue;
            eval_row row;
            row.prompt     = parts[0];
            row.answer     = parts[1];
            row.candidates = split(parts[2], '|');
            row.prefix     = parts.size() > 3 ? parts[3] : " ";
            rows.push_back(row);
        }
    }
    fprintf(stderr, "loaded %zu evaluation rows\n", rows.size());

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 512;
    cparams.n_batch   = 512;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { return 1; }

    llama_adapter_lora * lora = nullptr;
    if (!lora_path.empty()) {
        lora = llama_adapter_lora_init(model, lora_path.c_str());
        if (!lora) { return 1; }
        float scale = 1.0f;
        llama_set_adapters_lora(ctx, &lora, 1, &scale);
        fprintf(stderr, "lora applied: %s\n", lora_path.c_str());
    }

    llama_engram * engram = nullptr;
    if (!engram_path.empty()) {
        const int64_t t0 = ggml_time_us();
        engram = llama_engram_init_from_file(engram_path.c_str());
        if (!engram) { return 1; }
        llama_set_engram(ctx, engram);
        fprintf(stderr, "engram mounted in %.1f ms\n", (ggml_time_us() - t0) / 1000.0);
    }

    if (!gen_prompt.empty()) {
        llama_engram_signal(ctx, nullptr, nullptr, nullptr, true);
        const std::string completion = generate_greedy(ctx, vocab, gen_prompt, n_predict, engram != nullptr);
        if (engram) {
            float sig_mean = 0.0f, sig_max = 0.0f;
            int32_t sig_count = 0;
            llama_engram_signal(ctx, &sig_mean, &sig_max, &sig_count, false);
            printf("engram signal: mean=%.4f max=%.4f over %d tokens\n",
                    sig_mean, sig_max, sig_count);
        }
        printf("prompt: %s\ncompletion:%s\n%s\n", gen_prompt.c_str(), completion.c_str(),
                engram ? "[engram on]" : "[engram off]");
        if (rows.empty()) {
            if (engram) { llama_set_engram(ctx, nullptr); llama_engram_free(engram); }
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
    }

    int correct = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        const auto & row = rows[r];
        double best = -1e30;
        std::string picked;
        for (const auto & cand : row.candidates) {
            const double s = score_candidate(ctx, vocab, row.prompt, row.prefix + cand);
            if (s > best) { best = s; picked = cand; }
        }
        const bool ok = picked == row.answer;
        correct += ok ? 1 : 0;
        fprintf(stderr, "[%zu/%zu] %s picked='%s' answer='%s'\n",
                r + 1, rows.size(), ok ? "OK " : "MISS", picked.c_str(), row.answer.c_str());
    }

    printf("accuracy: %.4f (%d/%zu)%s\n",
            rows.empty() ? 0.0 : (double) correct / rows.size(),
            correct, rows.size(), engram ? " [engram on]" : " [engram off]");

    if (engram) {
        llama_set_engram(ctx, nullptr);
        llama_engram_free(engram);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
