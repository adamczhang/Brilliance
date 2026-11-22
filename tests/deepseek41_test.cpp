/* Bounded RAD2 fixture for the BF16-KV, exact-wire text graph. The real engine
 * runs the same file with one, two or four emulated devices on the host. */
#include "rad_test.h"
#include "e2e_config.h"
#include "tiny_model.h"

#include "build/startup.h"
#include "format/encoding.h"
#include "format/radfile.h"
#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"
#include "place/planner.h"
#include "text/chat.h"
#include "text/tokenizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <tuple>

namespace {
using namespace rad;
using namespace lq::ggml;

constexpr int kWidth = 256, kFfn = 1024, kExperts = 8, kLayers = 5, kVocab = 32;
constexpr int kEngramRows = 128, kEngramWidth = 64;
constexpr int kBatch = 16, kContext = 192;
/* The trace test's bound on the TP2-against-TP4 relative difference of any layer's MoE input: the
 * fixture measured 0.0162 at its worst layer (2026-11-22), about a third of this. */
constexpr double kTraceXfBound = 0.05;

const std::pair<const char*, const char*> kMeta[] = {
    { "qk_rope_head_dim", "64" }, { "q_lora_rank", "256" }, { "o_groups", "4" },
    { "o_lora_rank", "256" }, { "sliding_window", "128" },
    { "moe_intermediate_size", "1024" }, { "routed_scaling_factor", "1.5" },
    { "norm_topk_prob", "1" }, { "swiglu_limit", "10" },
    { "index_n_heads", "4" }, { "index_head_dim", "128" }, { "index_topk", "8" },
    { "candidate_source_layer_id", "3" }, { "candidate_topk_blocks", "2" },
    { "candidate_block_size", "8" }, { "hc_mult", "4" }, { "hc_sinkhorn_iters", "3" },
    { "hc_eps", "1e-6" }, { "compress_rope_theta", "160000" },
    { "compress_ratios", "0 2 2 1 1" }, { "kv_source_layer_ids", "1 3" },
    { "index_source_layer_ids", "1 3" }, { "engram_layer_ids", "0 2" },
    { "engram_max_ngram_size", "3" }, { "engram_n_heads", "2" },
    { "engram_head_dim", "64" }, { "engram_num_embeddings", "64 64" },
    { "engram_pad_token_id", "2" },
};

struct FixtureMeta {
    RadModelMeta meta{};
    std::vector<const char*> keys, values;

    FixtureMeta() {
        for (auto [key, value] : kMeta) {
            keys.push_back(key);
            values.push_back(value);
        }
        meta.arch_id = "deepseek41";
        meta.n_layers = kLayers;
        meta.n_embd = kWidth;
        meta.n_head = 4;
        meta.n_head_kv = 1;
        meta.head_dim = 512;
        meta.n_vocab = kVocab;
        meta.n_ctx_train = kContext;
        meta.n_expert = kExperts;
        meta.n_expert_used = 2;
        meta.n_expert_shared = 1;
        meta.rms_eps = 1e-6f;
        meta.rope_theta = 10000;
        meta.n_kv = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = values.data();
    }

    void write(RadWriter& writer) const {
        writer.set_arch(meta.arch_id);
        writer.set_model_name("deepseek41-text-fixture");
        writer.meta_i("n_layers", meta.n_layers);
        writer.meta_i("n_embd", meta.n_embd);
        writer.meta_i("n_head", meta.n_head);
        writer.meta_i("n_head_kv", meta.n_head_kv);
        writer.meta_i("head_dim", meta.head_dim);
        writer.meta_i("n_ff", kFfn);
        writer.meta_i("n_vocab", meta.n_vocab);
        writer.meta_i("n_ctx_train", meta.n_ctx_train);
        writer.meta_i("n_expert", meta.n_expert);
        writer.meta_i("n_expert_used", meta.n_expert_used);
        writer.meta_i("n_expert_shared", meta.n_expert_shared);
        writer.meta_f("rms_eps", meta.rms_eps);
        writer.meta_f("rope_theta", meta.rope_theta);
        for (auto [key, value] : kMeta) writer.meta_s(key, value);
    }
};

void encoded_weights(std::map<std::string, WeightSource>& sources,
                     std::map<std::string, ggml_type>& types) {
    auto add = [&](const std::string& name, ggml_type type, int rows, int cols) {
        auto& w = sources[name];
        w.enc = encoding_of(type);
        w.rank = 2;
        w.shape[0] = rows;
        w.shape[1] = cols;
        types.emplace(name, type);
    };
    add("token_embd.weight", GGML_TYPE_Q6_K, kVocab, kWidth);
    add("output.weight", GGML_TYPE_Q6_K, kVocab, kWidth);
    add("per_layer_token_embd.weight", GGML_TYPE_IQ4_NL, kEngramRows, kEngramWidth);
    for (int l = 0; l < kLayers; ++l) {
        const std::string layer = "blk." + std::to_string(l) + ".";
        add(layer + "ffn_gate_up_shexp.weight", GGML_TYPE_Q6_K, 2 * kFfn, kWidth);
        add(layer + "ffn_down_shexp.weight", GGML_TYPE_Q6_K, kWidth, kFfn);
        const ggml_type type = l % 2 ? GGML_TYPE_IQ2_XXS : GGML_TYPE_IQ1_S;
        for (int e = 0; e < kExperts; ++e) {
            const std::string suffix = "." + std::to_string(e) + ".weight";
            add(layer + "ffn_gate_up_exps" + suffix, type, 2 * kFfn, kWidth);
            add(layer + "ffn_down_exps" + suffix, type, kWidth, kFfn);
        }
    }
}

struct VocabSink : StringSink {
    explicit VocabSink(RadWriter& w) : writer(w) {}
    rad_stroff intern(std::string_view s) override { return writer.intern_string(s); }
    RadWriter& writer;
};

void write_vocab(RadWriter& writer) {
    VocabBuild vocab;
    vocab.tokens = { "<bos>", "<eos>", "<pad>" };
    for (char c = 'a'; c <= 'z'; ++c) vocab.tokens.emplace_back(1, c);
    vocab.tokens.insert(vocab.tokens.end(), { " ", "!", "." });
    vocab.types.assign(vocab.tokens.size(), RAD_TT_NORMAL);
    std::fill_n(vocab.types.begin(), 3, RAD_TT_CONTROL);
    vocab.bos = 0;
    vocab.eos = vocab.eot = 1;
    vocab.pad = 2;
    vocab.chat_template =
        "{% for message in messages %}{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}{% if enable_thinking %}<think>{% endif %}{% endif %}";
    VocabSink sink(writer);
    std::vector<uint8_t> section;
    REQUIRE_EQ(vocab_serialize(vocab, sink, 0, section), RAD_OK);
    REQUIRE_EQ(writer.set_vocab_section(section.data(), (int64_t)section.size()), RAD_OK);
}

void write_weight(RadWriter& writer, const RadWriter::Weight& w, uint32_t type, tiny::Rng& rng) {
    int64_t rows, cols;
    rad_enc_view(w.rank, w.shape, &rows, &cols);
    std::array<std::vector<uint8_t>, RAD_ENC_MAX_PLANES> planes;
    std::array<void*, RAD_ENC_MAX_PLANES> data{};
    std::array<int64_t, RAD_ENC_MAX_PLANES> pitch{};
    for (int p = 0; p < w.enc.n_planes; ++p) {
        int64_t pr, pc;
        const RadEncPlane& plane = w.enc.plane[p];
        rad_enc_plane_dims(&plane, rows, cols, &pr, &pc);
        pitch[p] = rad_dtype_bytes(plane.dtype, pc);
        planes[p].resize((size_t)(pr * pitch[p]));
        data[p] = planes[p].data();
    }
    if (const PlaneKind* kind = plane_kind(type)) {
        write_shared(kind->type, w.enc, data.data());
        const int64_t blocks = cols / kind->block;
        REQUIRE_EQ(cols % kind->block, 0);
        std::vector<uint8_t> native((size_t)blocks * kind->block_bytes);
        for (int64_t r = 0; r < rows; ++r) {
            for (uint8_t& b : native) b = (uint8_t)rng.next();
            for (int64_t j = 0; j < blocks; ++j) {
                uint8_t* block = native.data() + (size_t)j * kind->block_bytes;
                const uint16_t scale = type == GGML_TYPE_Q6_K ? 0x0400
                                     : type == GGML_TYPE_IQ4_NL ? 0x1800 : 0x1400;
                const size_t d = type == GGML_TYPE_Q6_K ? offsetof(block_q6_K, d) : 0;
                std::memcpy(block + d, &scale, sizeof scale);
            }
            std::array<uint8_t*, RAD_ENC_MAX_PLANES> views{};
            for (int p = 0; p < w.enc.n_planes; ++p)
                if (w.enc.plane[p].kind == RAD_PLANE_TILED && w.enc.plane[p].block[0] == 1)
                    views[p] = planes[p].data() + r * pitch[p];
            kind->unpack(native.data(), blocks, views.data());
            if (r == 0) {
                std::vector<uint8_t> packed(native.size());
                std::array<const uint8_t*, RAD_ENC_MAX_PLANES> inputs{};
                std::copy(views.begin(), views.end(), inputs.begin());
                REQUIRE(pack_of(type) != nullptr && dequant_of(type) != nullptr);
                pack_of(type)(inputs.data(), blocks, packed.data());
                CHECK(packed == native);
                std::array<const void*, RAD_ENC_MAX_PLANES> canonical{};
                std::copy(data.begin(), data.end(), canonical.begin());
                EncodedView encoded;
                enc_view(w.enc, rows, cols, canonical.data(), &encoded);
                std::vector<float> expected((size_t)cols), actual((size_t)cols);
                REQUIRE_EQ(enc_decode_rows(encoded, 0, 1, expected.data()), RAD_OK);
                dequant_of(type)(packed.data(), actual.data(), cols);
                CHECK(actual == expected);
            }
        }
    } else if (w.enc.plane[0].dtype == RAD_I32) {
        for (int32_t v = 0; v < kVocab; ++v)
            std::memcpy(planes[0].data() + v * sizeof v, &v, sizeof v);
    } else if (w.enc.plane[0].dtype == RAD_I64) {
        const int64_t values[] = { 3, 5, 7, 11 };
        for (int64_t i = 0; i < rows * cols; ++i) {
            int64_t value = values[i % 4];
            if (w.name.ends_with("engram_off"))
                value = (w.layer == 0 ? 0 : 64) + 16 * i;
            std::memcpy(planes[0].data() + i * sizeof value, &value, sizeof value);
        }
    } else if (w.name.ends_with("glm_router.correction")) {
        for (int i = 0; i < kExperts; ++i) {
            const float correction = i == w.layer ? 4.0f : i == (w.layer + 3) % kExperts ? 3.0f : 0.0f;
            rad_store_f32(data[0], RAD_F32, i, correction);
        }
    } else {
        const bool gain = w.name.find("norm.weight") != std::string::npos ||
                          w.name.find("_scale.weight") != std::string::npos ||
                          w.name.ends_with("engram_gate.weight");
        for (int64_t i = 0; i < rows * cols; ++i)
            rad_store_f32(data[0], w.enc.plane[0].dtype, i, gain ? 1.0f : 0.03f * rng.uniform());
    }
    std::array<const void*, RAD_ENC_MAX_PLANES> payload{};
    std::copy(data.begin(), data.end(), payload.begin());
    REQUIRE(writer.add_weight(w, payload.data()) >= 0);
}

void write_fixture(const std::string& path) {
    LoadedPlugins plugins;
    REQUIRE_EQ(rad_tools_load_plugins(RAD_E2E_HOME, { "libref" }, "deepseek41", "", &plugins), RAD_OK);
    std::map<std::string, WeightSource> sources;
    std::map<std::string, ggml_type> types;
    encoded_weights(sources, types);
    WeightSourceFn source = [&](const std::string& name, const Program&, WeightSource* out) {
        const auto it = sources.find(name);
        if (it == sources.end()) return RAD_E_NOTFOUND;
        *out = it->second;
        return RAD_OK;
    };
    const FixtureMeta meta;
    RadBuildCtx ctx{};
    ctx.world_size = 1;
    ctx.max_tok = kBatch;
    ctx.max_ctx = kContext;
    ctx.max_seqs = 1;
    ctx.kv_dtype = RAD_BF16;
    Program program;
    REQUIRE_EQ(rad_tools_declare(plugins, meta.meta, ctx, &program, {}, nullptr, source, false), RAD_OK);
    std::map<std::string, RadWriter::Weight> unique;
    for (size_t i = 1; i < program.weights.size(); ++i) {
        const auto& w = program.weights[i];
        auto [it, inserted] = unique.try_emplace(w.source);
        if (!inserted) continue;
        auto& stored = it->second;
        stored.name = w.source;
        stored.enc = w.enc;
        const auto encoded = sources.find(w.source);
        stored.rank = encoded == sources.end() ? w.decl.rank : encoded->second.rank;
        std::copy_n(encoded == sources.end() ? w.decl.shape : encoded->second.shape,
                    stored.rank, stored.shape);
        stored.layer = w.decl.group.layer;
        stored.expert = w.decl.group.expert;
        stored.slot = w.decl.group.slot;
    }
    std::vector<RadWriter::Weight> weights;
    for (auto& [name, w] : unique) weights.push_back(std::move(w));
    std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
        return std::tie(a.layer, a.expert, a.slot, a.name) < std::tie(b.layer, b.expert, b.slot, b.name);
    });
    RadWriter writer;
    REQUIRE_EQ(writer.begin(path.c_str()), RAD_OK);
    meta.write(writer);
    write_vocab(writer);
    tiny::Rng rng(41);
    for (const auto& w : weights) {
        const auto it = types.find(w.name);
        write_weight(writer, w, it == types.end() ? 0 : (uint32_t)it->second, rng);
    }
    REQUIRE_EQ(writer.finish(), RAD_OK);
}

/* The engine on the host backend over `kernels`: libref (the reference, which needs the debug
 * leave) or libavx alone, as a CPU-only deployment selects. */
void run_engine(const std::string& path, int world, int batch,
                const std::vector<std::string>& args, const std::string& log_path,
                const std::string& kernels = "libref",
                const std::vector<std::string>& extra_env = {}) {
    std::vector<std::string> command = {
        RAD_E2E_SERVER, "--model", path, "--kernels", kernels,
        "--tp", std::to_string(world), "--tp-wire", "exact", "--num-speculative-tokens", "0",
        "--kv-cache-dtype", "bf16", "--max-model-len", std::to_string(kContext),
        "--max-num-seqs", "1", "--max-num-batched-tokens", std::to_string(batch),
    };
    if (kernels == "libref") command.push_back("--debug-accept-reference-kernels");
    command.insert(command.end(), args.begin(), args.end());
    std::vector<std::string> env = {
        std::string("RADIANCE_HOME=") + RAD_E2E_HOME, "HIP_VISIBLE_DEVICES=-1",
        "ROCR_VISIBLE_DEVICES=-1", "CUDA_VISIBLE_DEVICES=-1",
        "RADIANCE_HOST_DEVICES=" + std::to_string(world), "RADIANCE_HOST_VRAM_MIB=512",
        "OMP_NUM_THREADS=2", "OMP_WAIT_POLICY=passive",
    };
    env.insert(env.end(), extra_env.begin(), extra_env.end());
    std::string log;
    const int rc = tiny::run(radtest::config_argv(command), env, &log, 80);
    CHECK(tiny::write_text(log_path, log));
    if (rc != 0) std::fprintf(stderr, "%s\n", log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(log.find("host backend") != std::string::npos);
    CHECK(log.find("arch=deepseek41") != std::string::npos);
}

float logp_distance(const std::string& a, const std::string& b) {
    if (a.empty() || a.size() != b.size() || a.size() % sizeof(uint16_t) != 0) return INFINITY;
    float worst = 0;
    for (size_t at = 0; at < a.size(); at += sizeof(uint16_t)) {
        uint16_t aa, bb;
        std::memcpy(&aa, a.data() + at, sizeof aa);
        std::memcpy(&bb, b.data() + at, sizeof bb);
        const float error = std::abs(f16_to_f32(aa) - f16_to_f32(bb));
        if (!std::isfinite(error)) return INFINITY;
        worst = std::max(worst, error);
    }
    return worst;
}

struct HeaderMeta {
    RadModelMeta meta{};
    std::vector<const char*> keys, values;
    std::vector<std::string> values_storage;

    explicit HeaderMeta(const RadFile& file) {
        meta.arch_id = file.str(file.header().arch_id);
        meta.name = file.str(file.header().model_name);
        meta.quant = file.str(file.header().quant);
        REQUIRE(file.get_i("n_layers", &meta.n_layers));
        REQUIRE(file.get_i("n_embd", &meta.n_embd));
        REQUIRE(file.get_i("n_head", &meta.n_head));
        REQUIRE(file.get_i("n_head_kv", &meta.n_head_kv));
        REQUIRE(file.get_i("head_dim", &meta.head_dim));
        REQUIRE(file.get_i("n_ff", &meta.n_ff));
        REQUIRE(file.get_i("n_vocab", &meta.n_vocab));
        REQUIRE(file.get_i("n_ctx_train", &meta.n_ctx_train));
        REQUIRE(file.get_i("n_expert", &meta.n_expert));
        REQUIRE(file.get_i("n_expert_used", &meta.n_expert_used));
        REQUIRE(file.get_i("n_expert_shared", &meta.n_expert_shared));
        double eps = 0, theta = 0;
        REQUIRE(file.get_f("rms_eps", &eps));
        REQUIRE(file.get_f("rope_theta", &theta));
        meta.rms_eps = (float)eps;
        meta.rope_theta = (float)theta;
        for (int64_t i = 0; i < file.meta_count(); ++i) {
            const auto& kv = file.meta_at(i);
            keys.push_back(file.str(kv.key));
            if (kv.type == RAD_P_INT) values_storage.push_back(fmt("%lld", (long long)kv.v.i));
            else if (kv.type == RAD_KV_F64) values_storage.push_back(fmt("%.17g", kv.v.f));
            else values_storage.emplace_back(file.str(kv.v.s));
        }
        for (const auto& value : values_storage) values.push_back(value.c_str());
        meta.n_kv = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = values.data();
    }
};
}  // namespace

TEST(deepseek41_rad2_binds_and_steps_on_mocked_tp1_tp2_tp4) {
    tiny::TempDir dir("deepseek41", std::getenv("BRILLIANCE_KEEP_DS41_FIXTURE") != nullptr);
    REQUIRE(dir.ok());
    const auto model = dir / "deepseek41.rad";
    write_fixture(model);
    RadFile file;
    REQUIRE_EQ(file.open(model.c_str()), RAD_OK);
    VocabBuild vocab_build;
    REQUIRE_EQ(vocab_deserialize(file.base(), file.size(), file.vocab().h, vocab_build), RAD_OK);
    Vocab vocab;
    REQUIRE_EQ(vocab.load(std::move(vocab_build)), RAD_OK);
    ChatTemplate chat;
    REQUIRE_EQ(chat.load_from_vocab(vocab), RAD_OK);
    const auto messages = Json::array({ { { "role", "user" }, { "content", "abcd" } } });
    for (bool thinking : { false, true }) {
        ChatOptions options;
        options.enable_thinking = thinking;
        ChatPrompt prompt;
        REQUIRE_EQ(chat.apply(messages, Json(), options, &prompt), RAD_OK);
        CHECK_EQ(prompt.prompt, std::string(thinking ? "abcd<think>" : "abcd"));
    }
    file.close();
    const auto corpus = dir / "corpus.jsonl";
    std::string prompt;
    for (int i = 0; i < 5; ++i) prompt += "abcd efghi jklmn opqr stuvwxyz!";
    REQUIRE(tiny::write_text(corpus, "{\"prompt\":\"" + prompt + "\"}\n"));
    std::string reference, token_ids;
    for (int world : { 1, 2, 4 }) {
        const auto out = dir / ("tp" + std::to_string(world));
        run_engine(model, world, kBatch, { "--kld-record", out, "--kld-corpus", corpus }, out + ".log");
        if (world == 1) {
            reference = tiny::read_text(out + "/logp.f16");
            token_ids = tiny::read_text(out + "/tokens.i32");
            const size_t tokens = token_ids.size() / sizeof(int32_t);
            REQUIRE(tokens > 128);
            CHECK_EQ(reference.size(), (tokens - 1) * kVocab * sizeof(uint16_t));
        } else {
            CHECK(token_ids == tiny::read_text(out + "/tokens.i32"));
            const float error = logp_distance(reference, tiny::read_text(out + "/logp.f16"));
            std::fprintf(stderr, "mocked TP%d maximum log-probability error %.6f\n", world, error);
            /* BF16 rounds each rank's partial projection before its exact-wire sum. The
             * chunk-size comparison below holds the rank geometry fixed and requires exactness. */
            CHECK(error <= 0.125f);
        }
    }
    const auto wide = dir / "tp1-wide";
    run_engine(model, 1, 32, { "--kld-record", wide, "--kld-corpus", corpus }, wide + ".log");
    CHECK(reference == tiny::read_text(wide + "/logp.f16"));
}

/* The same fixture on libavx alone, with no libref loaded and no leave to run reference kernels:
 * every op of the graph resolves to a CPU row (the DeepSeek-V4.1 ops of avx_ds41.cpp, avx_glm.cpp
 * and avx_ggml.cpp among them, the ggml rows over the weights' native blocks), or the server does
 * not start. Its log-probabilities are held to libref's within rounding: libavx differs only in
 * summation order, and the bf16 rounding of five layers turns that into a small difference. */
TEST(deepseek41_rad2_runs_on_libavx_alone_within_rounding_of_libref) {
#if defined(__x86_64__) || defined(_M_X64)
    NEED_ARTIFACT(RAD_E2E_HOME "/kernels/libavx.so");
#else
    SKIP("libavx is x86-64 only");
#endif
    tiny::TempDir dir("deepseek41-avx", std::getenv("BRILLIANCE_KEEP_DS41_FIXTURE") != nullptr);
    REQUIRE(dir.ok());
    const auto model = dir / "deepseek41.rad";
    write_fixture(model);
    const auto corpus = dir / "corpus.jsonl";
    std::string prompt;
    for (int i = 0; i < 5; ++i) prompt += "abcd efghi jklmn opqr stuvwxyz!";
    REQUIRE(tiny::write_text(corpus, "{\"prompt\":\"" + prompt + "\"}\n"));
    const auto ref = dir / "libref", avx = dir / "libavx";
    run_engine(model, 1, kBatch, { "--kld-record", ref, "--kld-corpus", corpus }, ref + ".log");
    run_engine(model, 1, kBatch, { "--kld-record", avx, "--kld-corpus", corpus }, avx + ".log",
               "libavx");
    const std::string avx_log = tiny::read_text(avx + ".log");
    /* libavx alone: the config echo names each loaded library at its version, and every op band
     * resolves to it. */
    CHECK(avx_log.find("libavx@") != std::string::npos);
    CHECK(avx_log.find("libref@") == std::string::npos);
    CHECK(avx_log.find("kernel selection                   libavx (host)") != std::string::npos);
    const std::string tokens = tiny::read_text(ref + "/tokens.i32");
    REQUIRE(tokens.size() / sizeof(int32_t) > 128);
    CHECK(tokens == tiny::read_text(avx + "/tokens.i32"));
    const float error = logp_distance(tiny::read_text(ref + "/logp.f16"),
                                      tiny::read_text(avx + "/logp.f16"));
    std::fprintf(stderr, "libavx against libref: maximum log-probability error %.6f\n", error);
    CHECK(error <= 0.125f);
}

/* ---- RADIANCE_DEBUG_DS41_DUMP's per-layer trace at TP2 and TP4 (ds41-tp2-trace, 2026-11-22).
 *
 * The real file's TP2 and TP4 disagreed on 10 of 96 teacher-forced top-1 tokens, two of them where
 * TP4 was confident. The trace located where the layouts part: each layer's streams, attention
 * partials and reduce, MoE input, routing and MoE output. This holds the trace and the invariants
 * the investigation relied on, on the fixture at TP2 and TP4:
 *  - every site of every layer, rank and prefill step is written, at its size;
 *  - the collectives: every rank holds the same reduced attention output, MoE input and MoE output,
 *    and the reduced attention output is the exact sum of the ranks' partials rounded once to bf16
 *    (within one bf16 ulp);
 *  - routing: each layout's experts are the top-k of sqrt(softplus(score)) + correction from the
 *    scores it computed, and the weights those scores' normalised values x 1.5;
 *  - layer 0's input is bit-identical on both layouts, and the MoE input of every layer stays
 *    within the bound below of the other layout's.
 * The fixture's corrections pin each layer's two experts, so it cannot reproduce a near-tie swap;
 * the real model's swaps are judged from a trace of the real file (the stage report). */
namespace {
struct Trace {
    /* (step, rows, layer, site, rank) -> path */
    std::map<std::tuple<int, int, int, std::string, int>, std::string> files;
    int ranks = 0;
};

Trace read_trace(const std::string& dir) {
    Trace t;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        int step = 0, rank = 0, layer = 0;
        long long rows = 0;
        char site[32] = {}, dt[8] = {};
        if (std::sscanf(n.c_str(), "s%d.t%lld.l%d.%31[a-z].r%d.%7s", &step, &rows, &layer, site,
                        &rank, dt) == 6) {
            t.files[{ step, (int)rows, layer, site, rank }] = e.path().string();
            t.ranks = std::max(t.ranks, rank + 1);
        }
    }
    return t;
}

std::vector<float> trace_f32(const Trace& t, int step, int rows, int layer, const char* site,
                             int rank, bool bf16) {
    const auto it = t.files.find({ step, rows, layer, site, rank });
    if (it == t.files.end()) return {};
    const std::string raw = tiny::read_text(it->second);
    std::vector<float> v;
    if (bf16) {
        v.resize(raw.size() / 2);
        for (size_t i = 0; i < v.size(); ++i) {
            uint16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            const uint32_t u = (uint32_t)h << 16;
            std::memcpy(&v[i], &u, 4);
        }
    } else {
        v.resize(raw.size() / 4);
        std::memcpy(v.data(), raw.data(), v.size() * 4);
    }
    return v;
}

std::vector<int32_t> trace_i32(const Trace& t, int step, int rows, int layer, int rank) {
    const auto it = t.files.find({ step, rows, layer, "ids", rank });
    if (it == t.files.end()) return {};
    const std::string raw = tiny::read_text(it->second);
    std::vector<int32_t> v(raw.size() / 4);
    std::memcpy(v.data(), raw.data(), v.size() * 4);
    return v;
}

double bf16_ulp(double x) {
    x = std::fabs(x);
    return x < 0x1p-126 ? 0x1p-133 : std::ldexp(1.0, (int)std::floor(std::log2(x)) - 7);
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        num += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        den += (double)b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0;
}
}  // namespace

TEST(deepseek41_layer_trace_holds_tp2_tp4_collectives_and_routing) {
    tiny::TempDir dir("deepseek41-trace", std::getenv("BRILLIANCE_KEEP_DS41_FIXTURE") != nullptr);
    REQUIRE(dir.ok());
    const auto model = dir / "deepseek41.rad";
    write_fixture(model);
    const auto corpus = dir / "corpus.jsonl";
    std::string prompt;
    for (int i = 0; i < 5; ++i) prompt += "abcd efghi jklmn opqr stuvwxyz!";
    REQUIRE(tiny::write_text(corpus, "{\"prompt\":\"" + prompt + "\"}\n"));
    const char* const sites[] = { "in", "attnp", "attn", "xf", "logits", "ids", "ew", "moe" };
    const int cols[] = { 4 * kWidth, kWidth, kWidth, kWidth, kExperts, 2, 2, kWidth };
    Trace tr[2];
    for (int w = 0; w < 2; ++w) {
        const int world = w ? 4 : 2;
        const auto out = dir / ("trace-tp" + std::to_string(world));
        const auto dump = out + "-dump";
        REQUIRE(std::filesystem::create_directories(dump));
        run_engine(model, world, kBatch, { "--kld-record", out, "--kld-corpus", corpus },
                   out + ".log", "libref", { "RADIANCE_DEBUG_DS41_DUMP=" + dump });
        tr[w] = read_trace(dump);
        CHECK_EQ(tr[w].ranks, world);
    }
    /* The prefill steps both layouts traced, by (step, rows). */
    std::set<std::pair<int, int>> steps[2];
    for (int w = 0; w < 2; ++w)
        for (const auto& [k, p] : tr[w].files) steps[w].insert({ std::get<0>(k), std::get<1>(k) });
    REQUIRE(!steps[0].empty());
    CHECK(steps[0] == steps[1]);
    double worst_xf = 0, worst_ar_ulp = 0;
    for (const auto& [step, T] : steps[0]) {
        for (int w = 0; w < 2; ++w) {
            const Trace& t = tr[w];
            for (int l = 0; l < kLayers; ++l) {
                /* Every site, every rank, at its size. */
                for (int s = 0; s < 8; ++s)
                    for (int r = 0; r < t.ranks; ++r) {
                        const auto it = t.files.find({ step, T, l, sites[s], r });
                        REQUIRE(it != t.files.end());
                        const int bytes = s >= 4 && s <= 6 ? 4 : 2;
                        CHECK_EQ((long long)std::filesystem::file_size(it->second),
                                 (long long)T * cols[s] * bytes);
                    }
                /* Every rank holds the same reduced rows. */
                for (const char* s : { "attn", "xf", "moe" }) {
                    const auto r0 = trace_f32(t, step, T, l, s, 0, true);
                    for (int r = 1; r < t.ranks; ++r) CHECK(trace_f32(t, step, T, l, s, r, true) == r0);
                }
                /* The reduce is the exact sum of the partials, rounded once. */
                std::vector<std::vector<float>> parts;
                for (int r = 0; r < t.ranks; ++r) parts.push_back(trace_f32(t, step, T, l, "attnp", r, true));
                const auto red = trace_f32(t, step, T, l, "attn", 0, true);
                REQUIRE_EQ(red.size(), (size_t)T * kWidth);
                for (size_t i = 0; i < red.size(); ++i) {
                    double sum = 0;   /* bf16 partials: the double sum is exact */
                    for (const auto& p : parts) sum += p[i];
                    worst_ar_ulp = std::max(worst_ar_ulp, std::fabs(red[i] - sum) / bf16_ulp(sum));
                }
                /* Routing from the layout's own scores: the fixture's correction (+4 on expert l,
                 * +3 on expert (l + 3) % 8) selects, the unbiased scores weigh. */
                const auto lg = trace_f32(t, step, T, l, "logits", 0, false);
                const auto ew = trace_f32(t, step, T, l, "ew", 0, false);
                const auto ids = trace_i32(t, step, T, l, 0);
                for (int row = 0; row < T; ++row) {
                    double sel[kExperts], sc[kExperts];
                    for (int e = 0; e < kExperts; ++e) {
                        const double x = lg[(size_t)row * kExperts + e];
                        sc[e] = std::sqrt(x > 20 ? x : std::log1p(std::exp(x)));
                        sel[e] = sc[e] + (e == l ? 4.0 : e == (l + 3) % kExperts ? 3.0 : 0.0);
                    }
                    int a = 0;
                    for (int e = 1; e < kExperts; ++e) if (sel[e] > sel[a]) a = e;
                    int b = a == 0 ? 1 : 0;
                    for (int e = 0; e < kExperts; ++e) if (e != a && sel[e] > sel[b]) b = e;
                    const int32_t i0 = ids[(size_t)row * 2], i1 = ids[(size_t)row * 2 + 1];
                    CHECK((i0 == a && i1 == b) || (i0 == b && i1 == a));
                    const double tot = sc[a] + sc[b];
                    const double wa = sc[a] / tot * 1.5, wb = sc[b] / tot * 1.5;
                    const double w0 = i0 == a ? wa : wb, w1 = i0 == a ? wb : wa;
                    CHECK(std::fabs(ew[(size_t)row * 2] - w0) <= 1e-5 * 1.5);
                    CHECK(std::fabs(ew[(size_t)row * 2 + 1] - w1) <= 1e-5 * 1.5);
                }
            }
        }
        /* The layouts against each other: layer 0's input exactly, then each MoE input and the
         * routing. */
        CHECK(trace_f32(tr[0], step, T, 0, "in", 0, true) == trace_f32(tr[1], step, T, 0, "in", 0, true));
        for (int l = 0; l < kLayers; ++l) {
            worst_xf = std::max(worst_xf, rel_l2(trace_f32(tr[0], step, T, l, "xf", 0, true),
                                                 trace_f32(tr[1], step, T, l, "xf", 0, true)));
            CHECK(trace_i32(tr[0], step, T, l, 0) == trace_i32(tr[1], step, T, l, 0));
        }
    }
    std::fprintf(stderr, "trace: %zu prefill steps; reduce within %.3f bf16 ulp of the exact sum; "
                         "worst TP2-TP4 MoE-input relative difference %.3g\n",
                 steps[0].size(), worst_ar_ulp, worst_xf);
    CHECK(worst_ar_ulp <= 1.0);
    CHECK(worst_xf <= kTraceXfBound);
}

/* Opt-in dry placement of the installed file. The weight blob is never read or
 * allocated. Reference layout sizes guide the plan without initializing kernels. */
TEST(deepseek41_bakon3_directory_reaches_dry_placement) {
    const char* path = std::getenv("BRILLIANCE_DS41_HEADER");
    if (!path || !*path) SKIP("BRILLIANCE_DS41_HEADER is unset (the installed model file, not an in-tree artifact)");
    RadFile file;
    REQUIRE_EQ(file.open(path), RAD_OK);
    REQUIRE_EQ(file.size(), INT64_C(269650269832));
    REQUIRE_EQ(file.entry_count(), 32701);
    const HeaderMeta header(file);
    LoadedPlugins plugins;
    REQUIRE_EQ(rad_tools_load_plugins(RAD_E2E_HOME, { "libref" }, header.meta.arch_id,
                                      header.meta.quant, &plugins), RAD_OK);
    const auto sources = rad_container_sources(file);
    for (int world : { 1, 2, 4 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuildCtx ctx{};
            ctx.rank = rank;
            ctx.world_size = world;
            ctx.max_tok = 512;
            ctx.max_ctx = 4096;
            ctx.max_seqs = 1;
            ctx.kv_dtype = RAD_BF16;
            Program program;
            REQUIRE_EQ(rad_tools_declare(plugins, header.meta, ctx, &program, {}, nullptr,
                                          sources, false), RAD_OK);
            int64_t expert_weights = 0;
            int engram_views = 0;
            for (rad_weight h = 1; h < (rad_weight)program.weights.size(); ++h) {
                const auto& w = program.weights[h];
                CHECK(w.enc_known);
                CHECK(file.find(w.source) != nullptr);
                CHECK(w.source.find("dspark.") == std::string::npos);
                CHECK(w.decl.group.layer < 40);
                expert_weights += w.decl.group.expert >= 0;
                engram_views += is_ngram_table(program, h);
            }
            CHECK_EQ(expert_weights, 2 * 40 * 384 / world);
            CHECK_EQ(engram_views, 3);
            Config cfg;
            cfg.tp = world;
            cfg.placement = "expert_tiered";
            cfg.vram_weights_mib = 28 * 1024;
            cfg.host_pool_mib = world == 4 ? 4 * 1024 : world == 2 ? 20 * 1024 : 0;
            cfg.ngram_placement = TablePlacement::Disk;
            Plan plan;
            const PlannerInput input{ &program, &cfg, rank, world, nullptr, 0, true, true };
            const int rc = Planner(input).plan(&plan);
            REQUIRE_EQ(plan.weight.size(), program.weights.size());
            std::fprintf(stderr, "TP%d rank %d dry placement: %d, %zu units, %zu unfit\n",
                         world, rank, rc, plan.units.size(), plan.unfit.size());
            for (rad_weight h = 1; h < (rad_weight)program.weights.size(); ++h)
                if (is_ngram_table(program, h)) CHECK(plan.weight[h].tier == Tier::Mapped);
        }
    }
}

/* Brilliance: the installed file's TP2 dry plan with the routed-expert file and RAM tiers on
 * (--weights-disk-tier, --expert-ram-mib 24576). The plan the refusal above leaves unfit fits:
 * the experts it spills are whole, unsharded and relaid from their IQ1_S/IQ2_XXS planes, so the
 * file tier takes them, and the RAM store holds them hottest first within each rank's 12 GiB. */
TEST(deepseek41_bakon3_tp2_spills_relaid_experts_to_the_ram_tier) {
    const char* path = std::getenv("BRILLIANCE_DS41_HEADER");
    if (!path || !*path) SKIP("BRILLIANCE_DS41_HEADER is unset (the installed model file, not an in-tree artifact)");
    RadFile file;
    REQUIRE_EQ(file.open(path), RAD_OK);
    const HeaderMeta header(file);
    LoadedPlugins plugins;
    REQUIRE_EQ(rad_tools_load_plugins(RAD_E2E_HOME, { "libref" }, header.meta.arch_id,
                                      header.meta.quant, &plugins), RAD_OK);
    const auto sources = rad_container_sources(file);
    const int world = 2;
    for (int rank = 0; rank < world; ++rank) {
        RadBuildCtx ctx{};
        ctx.rank = rank;
        ctx.world_size = world;
        ctx.max_tok = 512;
        ctx.max_ctx = 4096;
        ctx.max_seqs = 1;
        ctx.kv_dtype = RAD_BF16;
        Program program;
        REQUIRE_EQ(rad_tools_declare(plugins, header.meta, ctx, &program, {}, nullptr, sources,
                                      false), RAD_OK);
        Config cfg;
        cfg.tp = world;
        cfg.placement = "expert_tiered";
        cfg.vram_weights_mib = 28 * 1024;
        cfg.host_pool_mib = 20 * 1024;
        cfg.ngram_placement = TablePlacement::Disk;
        cfg.weights_disk_tier = true;
        cfg.weights_file_relayout = true;
        cfg.expert_ram_mib = 12 * 1024;
        Plan plan;
        /* libref alone, as above: every op is host-only, so the host backend keeps the relaid
         * experts out of the container's mapping and the pools and the file tier place them. */
        const PlannerInput input{ &program, &cfg, rank, world, nullptr, 0, true, true };
        const int rc = Planner(input).plan(&plan);
        std::fprintf(stderr, "TP2 rank %d with the RAM tier: %d, %zu unfit\n  %s\n", rank, rc,
                     plan.unfit.size(), plan.tier_line().c_str());
        CHECK_EQ(rc, (int)RAD_OK);
        CHECK(plan.unfit.empty());
        int64_t ssd = 0, relaid = 0;
        for (const MoveUnit& mu : plan.units) {
            if (mu.tier != Tier::SSD) continue;
            ++ssd;
            CHECK(mu.access == RAD_ACCESS_CONDITIONAL && mu.expert >= 0);
            relaid += mu.file_relaid;
        }
        CHECK(ssd > 0);
        CHECK_EQ(relaid, ssd);
        CHECK(plan.ram_units > 0);
        CHECK(plan.ram_bytes <= plan.ram_budget);
        CHECK_EQ(plan.ram_units + plan.disk_units, ssd);
    }
}

RAD_TEST_MAIN()
