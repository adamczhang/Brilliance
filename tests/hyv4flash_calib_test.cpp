/* Hy4-flash's calibration taps (arch/hyv4flash/hyv4flash_calib.h) on the declared mini graph, without
 * a GPU or checkpoint: which Hessian files a `--calib-dir` run records and which weights each
 * names; that every quantisable trunk linear is named by exactly one site, with the width its
 * declared weight reads; `--calib-layers`; the CLI-to-environment mapping the experts' tap reads;
 * a step issuing one Gram a site; and tools/calib_st2gram.py turning a site file into the
 * `gram.r0.<weight stem>.bin` a recipe's `gptq calib=` looks for (libquant/lq_hess.cpp). */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "rad_plugin.h"
#include "../arch/hyv4flash/hyv4flash.cpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include <unistd.h>

using namespace rad_fake;

namespace {

/* The `mini` preset as rad-convert records its config: 3 layers (sparse, sliding, sparse; layer 0
 * dense, 1-2 latent MoE), two streams, an Engram in front of layer 1, dense limit 16. */
MetaKV mini_meta() {
    RadModelMeta m{};
    m.arch_id = "hyv4flash";
    m.name = "test-hy4-mini";
    m.quant = "";
    m.n_layers = 3;
    m.n_embd = 128;
    m.n_head = 2;
    m.n_head_kv = 2;
    m.head_dim = 8;
    m.n_ff = 256;
    m.n_vocab = 320;
    m.n_ctx_train = 64;
    m.rms_eps = 1e-5f;
    m.rope_theta = 10000.0f;
    m.n_expert = 4;
    m.n_expert_used = 2;
    MetaKV k(m);
    const char* kv[][2] = {
        { "hc_mult", "2" }, { "q_lora_rank", "64" }, { "kv_lora_rank", "32" },
        { "num_attention_heads", "2" }, { "qk_nope_head_dim", "24" }, { "qk_rope_head_dim", "8" },
        { "v_head_dim", "32" }, { "swa_num_attention_heads", "4" },
        { "swa_qk_nope_head_dim", "24" }, { "swa_qk_rope_head_dim", "8" },
        { "swa_v_head_dim", "32" }, { "sliding_window", "8" },
        { "layer_types", "deepseek_sparse_attention sliding_attention deepseek_sparse_attention" },
        { "mlp_layer_types", "dense sparse sparse" }, { "intermediate_size", "256" },
        { "n_routed_experts", "4" }, { "n_shared_experts", "1" }, { "num_experts_per_tok", "2" },
        { "moe_intermediate_size", "128" }, { "routed_expert_hidden_size", "128" },
        { "routed_scaling_factor", "3.998" }, { "swiglu_limit", "10" }, { "norm_topk_prob", "1" },
        { "index_n_heads", "2" }, { "index_head_dim", "16" },
        { "compressed_indexer_compress_ratio", "4" }, { "compressed_indexer_group_topk", "4" },
        { "use_mla", "1" }, { "gated_mla", "1" }, { "learnable_sink", "1" },
        { "sandwich_norm", "1" }, { "enable_ihc", "1" }, { "use_dsa", "1" },
        { "use_compressed_indexer", "1" }, { "attention_bias", "0" },
        { "moe_router_use_sigmoid", "1" }, { "moe_router_enable_expert_bias", "1" },
        { "hc_eps", "1e-06" }, { "hc_magnitude", "2" },
        { "rope_parameters.sliding_attention.rope_theta", "10000" },
        { "rope_parameters.deepseek_sparse_attention.rope_theta", "10000000" },
        { "engram_layer_ids", "2" }, { "engram_max_ngram_size", "3" },
        { "engram_n_head_per_ngram", "5" }, { "engram_n_embed_per_ngram", "40" },
        { "engram_layer_head_vocab_sizes.2", "11 13 17 19 23 29 31 37 41 43" },
        { "engram_pad_id", "2" }, { "engram_lookup_table_size", "300" },
        { "engram_fp8_embedding", "1" }, { "image_token_id", "120120" },
        { "video_token_id", "120124" },
    };
    for (const auto& p : kv) k.set(p[0], p[1]);
    return k;
}

/* A bf16 container: the experts plain (MoeFP8 reads the format off the first expert). */
void bf16_encs(RadBuilder* b) {
    b->encs.clear();
    b->encs.push_back({ "ffn_gate_up_exps", rad_enc_plain(RAD_BF16) });
    b->encs.push_back({ "ffn_down_exps", rad_enc_plain(RAD_BF16) });
    b->kv_block_by_consumer = true;
}


struct Declared {
    MetaKV mk = mini_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    int rc = 0;
    /* `lo:hi` 0:0 records every layer. */
    explicit Declared(const char* dir, int lo = 0, int hi = 0, long long tokens = 100, int world = 1,
                      int rank = 0) {
        bf16_encs(&b);
        ctx = build_ctx(rank, world);
        ctx.calib_dir = dir;
        ctx.calib_tokens = tokens;
        ctx.calib_layer_lo = lo;
        ctx.calib_layer_hi = hi;
        rc = stderr_of_rc([&] { return hyv4flash::declare(&b, mk.meta(), &ctx); });
    }
    const hyv4flash::Model& model() const { return *hyv4flash::models[rad_build_slot(&ctx)]; }
};

/* A site's stem and its linears. */
std::map<std::string, std::string> sites_of(const hyv4flash::Model& m) {
    std::map<std::string, std::string> r;
    for (const auto& s : m.calib.cf.sites) r[s.stem] = s.linears;
    return r;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> r;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) r.push_back(t);
    return r;
}

/* The environment the experts' tap reads is process-wide; each test starts clean. */
void clean_env() {
    unsetenv("RADIANCE_CALIB_DIR");
    unsetenv("RADIANCE_CALIB_TOKENS");
    unsetenv("RADIANCE_CALIB_LAYERS");
}

}  // namespace

/* PLAN §4.1's table on the mini graph: layer 0 (sparse, dense MLP), layer 1 (sliding, MoE, the
 * Engram in front), layer 2 (sparse, MoE), and the head. */
TEST(the_sites_are_the_plans_table) {
    clean_env();
    Declared d("/calib");
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto s = sites_of(d.model());
    const std::map<std::string, std::string> want = {
        { "blk.0.attn_q_a.input", "blk.0.attn_q_a,blk.0.attn_kv_a,blk.0.attn_gate,blk.0.index_key,blk.0.index_head" },
        { "blk.0.attn_q_b.input", "blk.0.attn_q_nope,blk.0.attn_q_rope,blk.0.index_query_nope,blk.0.index_query_rope" },
        { "blk.0.attn_kv_b.input", "blk.0.attn_kv_b" },
        { "blk.0.attn_output.input", "blk.0.attn_output" },
        { "blk.0.ffn_gate_up.input", "blk.0.ffn_gate_up" },
        { "blk.0.ffn_down.input", "blk.0.ffn_down" },
        { "blk.1.attn_q_a.input", "blk.1.attn_q_a,blk.1.attn_kv_a,blk.1.attn_gate" },
        { "blk.1.attn_q_b.input", "blk.1.attn_q_nope,blk.1.attn_q_rope" },
        { "blk.1.attn_kv_b.input", "blk.1.attn_kv_b" },
        { "blk.1.attn_output.input", "blk.1.attn_output" },
        { "blk.1.ffn_latent_in.input", "blk.1.ffn_latent_in,blk.1.ffn_gate_up_shexp" },
        { "blk.1.ffn_down_shexp.input", "blk.1.ffn_down_shexp" },
        { "blk.1.ffn_latent_out.input", "blk.1.ffn_latent_out" },
        { "blk.1.engram_key.input", "blk.1.engram_key,blk.1.engram_value" },
        { "blk.2.attn_q_a.input", "blk.2.attn_q_a,blk.2.attn_kv_a,blk.2.attn_gate,blk.2.index_key,blk.2.index_head" },
        { "blk.2.attn_q_b.input", "blk.2.attn_q_nope,blk.2.attn_q_rope,blk.2.index_query_nope,blk.2.index_query_rope" },
        { "blk.2.attn_kv_b.input", "blk.2.attn_kv_b" },
        { "blk.2.attn_output.input", "blk.2.attn_output" },
        { "blk.2.ffn_latent_in.input", "blk.2.ffn_latent_in,blk.2.ffn_gate_up_shexp" },
        { "blk.2.ffn_down_shexp.input", "blk.2.ffn_down_shexp" },
        { "blk.2.ffn_latent_out.input", "blk.2.ffn_latent_out" },
        { "output.input", "output" },
    };
    CHECK_EQ(s.size(), want.size());
    for (const auto& [stem, lin] : want) {
        auto it = s.find(stem);
        CHECK(it != s.end());
        if (it == s.end()) continue;
        if (it->second != lin) fprintf(stderr, "%s: '%s', want '%s'\n", stem.c_str(), it->second.c_str(), lin.c_str());
        CHECK(it->second == lin);
    }
}

/* libquant finds a Gram by the weight's own stem, so every linears entry must be a declared weight
 * `<stem>.weight` whose input width is the site's, and every quantisable trunk linear must be in
 * exactly one site. Not linears a recipe quantises: the router (full-precision logits), norms and parameters (f32 or
 * one-dimensional), the indexer's pooling projections (f32), the Engram table and the experts
 * (their own tap). */
TEST(every_linears_entry_is_a_declared_weight_of_the_sites_width_and_every_trunk_linear_has_a_site) {
    clean_env();
    Declared d("/calib");
    REQUIRE_EQ(d.rc, RAD_OK);
    std::map<std::string, int> covered;
    for (const auto& site : d.model().calib.cf.sites)
        for (const std::string& lin : split(site.linears)) {
            const RecWeight* w = find_w(d.b, lin + ".weight");
            if (!w) fprintf(stderr, "site %s names '%s', which is not a declared weight\n", site.stem.c_str(), lin.c_str());
            CHECK(w != nullptr);
            if (!w) continue;
            CHECK_EQ(w->d.shape[1], site.K);
            ++covered[lin];
        }
    for (const auto& [lin, n] : covered) {
        if (n != 1) fprintf(stderr, "%s is named by %d sites\n", lin.c_str(), n);
        CHECK_EQ(n, 1);
    }
    for (const auto& w : d.b.weights) {
        const std::string& n = w.name;
        if (n.compare(0, 4, "blk.") != 0 && n != "output.weight") continue;
        if (w.d.dtype != RAD_BF16 || w.d.shape[1] <= 0) continue;   /* f32 parameters, 1-D */
        if (n.ends_with(".glm_router.weight")) continue;
        if (n.find("_exps.") != std::string::npos || n.find("engram_table") != std::string::npos) continue;
        const std::string stem = n.substr(0, n.size() - 7);
        if (!covered.count(stem)) fprintf(stderr, "linear %s has no calibration site\n", n.c_str());
        CHECK(covered.count(stem) == 1);
    }
}

/* At two ranks the attention output's, the dense down's and the shared down's inputs are split by
 * columns (the others whole), one rank owns each layer, and the same files are named on every rank. */
TEST(column_split_inputs_are_gathered_at_two_ranks) {
    clean_env();
    std::map<std::string, std::string> first;
    for (int rank : { 0, 1 }) {
        Declared d("/calib", 0, 0, 100, 2, rank);
        REQUIRE_EQ(d.rc, RAD_OK);
        std::map<std::string, std::string> names = sites_of(d.model());
        if (rank == 0) first = names;
        CHECK(names == first);
        for (const auto& s : d.model().calib.cf.sites) {
            const bool split = s.stem.find("attn_output") != std::string::npos ||
                               s.stem.find(".ffn_down.") != std::string::npos ||
                               s.stem.find("ffn_down_shexp") != std::string::npos;
            CHECK_EQ(s.k_local * (split ? 2 : 1), s.K);
            CHECK_EQ(s.op_gather != 0, split);
            const int layer = s.stem == "output.input" ? 0 : std::atoi(s.stem.c_str() + 4);
            CHECK_EQ(s.owner, layer % 2 == rank);
        }
    }
    clean_env();
}

/* --calib-layers LO:HI records those layers; the head's input only with the run that ends at the
 * last layer. No --calib-dir: no site and no scratch. */
TEST(calib_layers_and_no_calibration) {
    clean_env();
    {
        Declared d("/calib", 1, 2);
        REQUIRE_EQ(d.rc, RAD_OK);
        for (const auto& [stem, lin] : sites_of(d.model())) CHECK(stem.compare(0, 6, "blk.1.") == 0);
        CHECK(!sites_of(d.model()).empty());
        CHECK_EQ(std::string(std::getenv("RADIANCE_CALIB_LAYERS")), std::string("1-1"));
        /* The experts' tap only inside the window: layer 1's, not layer 2's. */
        CHECK(find_b(d.b, "blk.1.gram_dn_experts") != nullptr);
        CHECK(find_b(d.b, "blk.2.gram_gu") == nullptr);
        CHECK(find_b(d.b, "blk.2.gram_dn_experts") == nullptr);
    }
    clean_env();
    {
        Declared d("/calib", 2, 3);
        REQUIRE_EQ(d.rc, RAD_OK);
        CHECK(sites_of(d.model()).count("output.input") == 1);
        CHECK(sites_of(d.model()).count("blk.2.attn_q_a.input") == 1);
        CHECK(sites_of(d.model()).count("blk.1.attn_q_a.input") == 0);
        CHECK(find_b(d.b, "blk.2.gram_dn_experts") != nullptr);
        CHECK(find_b(d.b, "blk.1.gram_gu") == nullptr);
    }
    clean_env();
    {
        Declared d(nullptr);
        REQUIRE_EQ(d.rc, RAD_OK);
        CHECK(sites_of(d.model()).empty());
        CHECK(find_b(d.b, "calib.scratch") == nullptr);
        CHECK(std::getenv("RADIANCE_CALIB_DIR") == nullptr);
    }
}

/* The experts' tap is environment-driven: the CLI values reach it, and a variable naming another
 * directory is refused rather than splitting one run's Grams over two places. */
TEST(the_cli_values_reach_the_experts_tap) {
    clean_env();
    {
        Declared d("/calib", 0, 0, 4321);
        REQUIRE_EQ(d.rc, RAD_OK);
        CHECK_EQ(std::string(std::getenv("RADIANCE_CALIB_DIR")), std::string("/calib"));
        CHECK_EQ(std::string(std::getenv("RADIANCE_CALIB_TOKENS")), std::string("4321"));
    }
    clean_env();
    setenv("RADIANCE_CALIB_DIR", "/elsewhere", 1);
    CHECK_EQ(Declared("/calib").rc, RAD_E_INVAL);
    clean_env();
}

/* A prefill step issues one Gram a recorded input, the lint holds, and the experts' blocks declare
 * their own taps (gate_up pooled, down pooled and a Gram per expert). */
TEST(a_prefill_step_issues_one_gram_a_site_and_stays_lint_clean) {
    clean_env();
    Declared d("/calib");
    REQUIRE_EQ(d.rc, RAD_OK);
    for (const char* n : { "blk.1.gram_gu", "blk.1.gram_dn", "blk.2.gram_gu", "blk.2.gram_dn" })
        CHECK(find_b(d.b, n) != nullptr);
    for (const auto& pq : { std::vector<int>{ 12 }, std::vector<int>{ 40 } }) {
        auto k = kit(d.b, {}, pq, 0);
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(d.b, &c);
        if (!c.step_fail.empty()) fprintf(stderr, "%s\n", c.step_fail.c_str());
        CHECK(c.step_fail.empty());
        /* The trunk's sites, and the two MoE layers' pooled gate_up and down Grams. */
        CHECK_EQ((size_t)count_issued(issued_ops(d.b, c), "gram_accum"),
                 d.model().calib.cf.sites.size() + 2 * 2);
    }
    clean_env();
}

/* tools/calib_st2gram.py: a site file in CalibFull's layout becomes a RADGRAM2 file under every
 * stem of its `linears`, the whole symmetric matrix, rows and domain carried. */
TEST(calib_st2gram_writes_a_radgram2_file_under_each_weight_stem) {
#ifdef RAD_CALIB_ST2GRAM
    namespace fs = std::filesystem;
    /* Brilliance: the script needs numpy, which a Windows Python may not have. */
#if defined(_WIN32)
    if (std::system("python3 -I -c \"import numpy\" > NUL 2>&1") != 0) SKIP("numpy is not installed");
#else
    if (std::system("python3 -I -c 'import numpy' > /dev/null 2>&1") != 0) SKIP("numpy is not installed");
#endif
    /* Brilliance: under the test's own directory (RAD_TEST_TMPDIR, rad_test.h), not /tmp. */
    const std::string dir = radtest::tmp_path("hy4_calib");
    fs::create_directories(dir);
    const int64_t k = 6;
    std::vector<float> tri;
    std::vector<float> full((size_t)(k * k));
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = i; j < k; ++j) {
            const float v = (float)(10 * i + j) + 0.25f;
            tri.push_back(v);
            full[(size_t)(i * k + j)] = full[(size_t)(j * k + i)] = v;
        }
    {
        std::string hdr = "{\"__metadata__\":{\"k\":\"6\",\"rows\":\"77\",\"domain\":\"128\","
                          "\"linears\":\"blk.3.attn_q_a,blk.3.attn_kv_a\"},\"hessian_upper\":"
                          "{\"dtype\":\"F32\",\"shape\":[21],\"data_offsets\":[0,84]}}";
        while (hdr.size() % 8) hdr += ' ';
        const uint64_t hl = hdr.size();
        std::ofstream f(dir + "/blk.3.attn_q_a.input.safetensors", std::ios::binary);
        f.write((const char*)&hl, 8);
        f.write(hdr.data(), (std::streamsize)hl);
        f.write((const char*)tri.data(), (std::streamsize)(tri.size() * 4));
    }
    /* Quoted: the tree's path may hold a space (Brilliance's does). cmd.exe strips one outer pair
     * of quotes when the line starts with one, so the line starts with the bare interpreter. */
    const std::string cmd = std::string("python3 -I \"") + RAD_CALIB_ST2GRAM + "\" \"" + dir + "\" > \"" +
                            dir + "/log\" 2>&1";
    const int st = std::system(cmd.c_str());
    if (st != 0) fprintf(stderr, "calib_st2gram.py exited %d (python3 with numpy needed):\n%s\n", st,
                         std::string(std::istreambuf_iterator<char>(std::ifstream(dir + "/log").rdbuf()), {}).c_str());
    REQUIRE_EQ(st, 0);
    for (const char* stem : { "blk.3.attn_q_a", "blk.3.attn_kv_a" }) {
        std::ifstream f(dir + "/gram.r0." + stem + ".bin", std::ios::binary);
        REQUIRE(f.good());
        char h[32];
        f.read(h, 32);
        CHECK(std::memcmp(h, "RADGRAM2", 8) == 0);
        uint32_t kk, rank, dom;
        uint64_t rows;
        std::memcpy(&kk, h + 8, 4);
        std::memcpy(&rank, h + 12, 4);
        std::memcpy(&rows, h + 16, 8);
        std::memcpy(&dom, h + 24, 4);
        CHECK_EQ((int64_t)kk, k);
        CHECK_EQ(rank, 0u);
        CHECK_EQ(rows, (uint64_t)77);
        CHECK_EQ(dom, 128u);
        std::vector<float> got((size_t)(k * k));
        f.read((char*)got.data(), (std::streamsize)(got.size() * 4));
        CHECK(got == full);
    }
    fs::remove_all(dir);
#endif
}

RAD_TEST_MAIN()
