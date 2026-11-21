/* deepseek41 -- DeepSeek-V4.1 (GGUF `general.architecture: deepseek41`, the layout llama.cpp's
 * converter writes): `hc` hyper-connection residual streams over latent attention layers that
 * attend a sliding window of raw keys plus, past the first layers, a two-level sparse top-k of
 * compressed keys; sqrt(softplus)-routed MoE with one shared expert; Engram n-gram lookups written
 * into the streams at a few layers. Every dimension comes from the checkpoint's metadata, so the
 * same graph serves the released model and any reduced one (tiny/ds41-tiny.gguf in the port).
 *
 * The numerics follow the official inference/model.py (DeepSeek's release): the window keys, the
 * compressed keys and the index keys are quantised and dequantised exactly where the reference
 * does it (quant_roundtrip), so the bf16 caches hold the reference's values.
 *
 * Per layer, in issue order:
 *   engram (at ple layers)   token_map -> ngram_ids -> embed_lookup -> cast to the card -> gemm ->
 *                            stream_gate_add (every layer's ngram_ids issued before the first
 *                            block, see step())
 *   attention connection     mhc_mix (this layer's gates) ; mhc_collapse (the previous gates)
 *   attention                q path, window key into the window cache, the compressor at a
 *                            kv-source layer (qsa_work / softmax_pool, or one entry a token at
 *                            ratio 1), the indexer at an index-source layer, attn_sparse_sink,
 *                            the inverse rotation, the grouped output projection ; mhc_post
 *   ffn connection           mhc_mix ; mhc_collapse ; MoeFP8 (norm, router, experts) ; mhc_post
 * then mhc_collapse with the last gates, the final norm and the head.
 *
 */
#include <arch/rad_block_moe_fp8.h>
#include <arch/rad_deepseek41_chat.h>
#include <arch/rad_expert_split.h>
#include <rad_kv_fp8.h>


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace deepseek41 {
using namespace rad::arch;

/* RAD_ARCH_TRY that names the call that failed: a refused declaration otherwise surfaces only as
 * the plugin's return code. */
#define DS_TRY(e)                                                                                \
    do {                                                                                         \
        const int rc_ = (e);                                                                     \
        if (rc_ < 0) {                                                                           \
            fprintf(stderr, "radiance: deepseek41: %s refused (%d)\n", #e, rc_);                  \
            return rc_;                                                                          \
        }                                                                                        \
    } while (0)

/* ------------------------------------------------------------------ the configuration */
struct Cfg {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, head_dim = 0, rope = 0, q_lora = 0;
    int64_t o_groups = 0, o_rank = 0, window = 0;
    int64_t n_expert = 0, top_k = 0, n_ff = 0, n_shared = 0;
    double  route_scale = 1.0, swiglu_limit = 0.0;
    int     norm_topk = 1;
    int64_t ih = 0, idim = 0, itopk = 0, cand = -1, cand_blocks = 0, cand_block = 0;
    int64_t hc = 0, hc_iters = 0;
    double  hc_eps = 0, eps = 0;
    double  theta = 0, comp_theta = 0, yarn_factor = 0, yarn_low = 0, yarn_high = 0;
    std::vector<int64_t> ratio, kv_src, idx_src, ple;
    int64_t ngram = 0, ple_heads = 0, ple_dim = 0, ple_rows = 0, pad = 0;
    int64_t n_vocab = 0;
    /* derived, per layer */
    std::vector<int> src_of, idx_of;
    bool is_src(int l) const { for (int64_t s : kv_src) if (s == l) return true; return false; }
    bool is_idx(int l) const { for (int64_t s : idx_src) if (s == l) return true; return false; }
    int  ple_index(int l) const {
        for (size_t i = 0; i < ple.size(); ++i) if (ple[i] == l) return (int)i;
        return -1;
    }
};

static std::string key(const char* k) { return std::string("deepseek41.") + k; }
/* DeepSeek's release states the same values in its config.json (text_config's keys, which the
 * core also reads bare). Each deepseek41.* key read here and the release's name for it; a key the
 * release does not state (the Engram's bucket sizes, derived from its primes) has none. */
static const char* release_key(const char* k) {
    static const struct { const char* gguf; const char* hf; } t[] = {
        { "rope.dimension_count", "qk_rope_head_dim" },
        { "attention.q_lora_rank", "q_lora_rank" },
        { "attention.output_group_count", "o_groups" },
        { "attention.output_lora_rank", "o_lora_rank" },
        { "attention.sliding_window", "sliding_window" },
        { "expert_feed_forward_length", "moe_intermediate_size" },
        { "expert_weights_scale", "routed_scaling_factor" },
        { "expert_weights_norm", "norm_topk_prob" },
        { "swiglu_clamp_exp", "swiglu_limit" },
        { "attention.indexer.head_count", "index_n_heads" },
        { "attention.indexer.key_length", "index_head_dim" },
        { "attention.indexer.top_k", "index_topk" },
        { "attention.indexer.candidate_source_layer", "candidate_source_layer_id" },
        { "attention.indexer.candidate_top_k_blocks", "candidate_topk_blocks" },
        { "attention.indexer.candidate_block_size", "candidate_block_size" },
        { "hyper_connection.count", "hc_mult" },
        { "hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters" },
        { "hyper_connection.epsilon", "hc_eps" },
        { "attention.compress_rope_freq_base", "compress_rope_theta" },
        { "attention.compress_ratios", "compress_ratios" },
        { "attention.kv_source_layers", "kv_source_layer_ids" },
        { "attention.indexer.source_layers", "index_source_layer_ids" },
        { "ple.layers", "engram_layer_ids" },
        { "ple.ngram_size", "engram_max_ngram_size" },
        { "ple.heads_per_ngram", "engram_n_heads" },
        { "embedding_length_per_layer_input", "engram_head_dim" },
        { "engram.pad_token_id", "engram_pad_token_id" }
    };
    for (const auto& e : t)
        if (std::strcmp(e.gguf, k) == 0) return e.hf;
    return nullptr;
}
/* deepseek41.<k>, else the release's spelling of it; null when neither is there. */
static const char* gs(const RadModelMeta* m, const char* k) {
    const char* s = rad_meta_gets(m, key(k).c_str(), nullptr);
    if (!s)
        if (const char* r = release_key(k)) s = rad_meta_gets(m, r, nullptr);
    return s;
}
static int64_t gi(const RadModelMeta* m, const char* k, int64_t d) {
    const char* s = gs(m, k);
    return s && *s ? std::strtoll(s, nullptr, 10) : d;
}
static double gf(const RadModelMeta* m, const char* k, double d) {
    const char* s = gs(m, k);
    return s && *s ? std::strtod(s, nullptr) : d;
}
/* An integer array spelled out, comma-separated (core/format/checkpoint.cpp, meta_from_gguf) or
 * space-separated (a config.json array). */
static std::vector<int64_t> int_list(const char* s) {
    std::vector<int64_t> v;
    if (!s || !*s || *s == '[') return v;
    while (*s) {
        char* e = nullptr;
        v.push_back(std::strtoll(s, &e, 10));
        if (e == s) return {};
        s = *e == ',' ? e + 1 : e;
    }
    return v;
}
static std::vector<int64_t> gl(const RadModelMeta* m, const char* k) { return int_list(gs(m, k)); }

static int read_cfg(Cfg& c, const RadModelMeta* m) {
    c.n_layer = m->n_layers;
    c.n_embd = m->n_embd;
    c.n_head = m->n_head;
    c.head_dim = m->head_dim;
    c.n_vocab = m->n_vocab;
    c.rope = gi(m, "rope.dimension_count", 0);
    c.q_lora = gi(m, "attention.q_lora_rank", 0);
    c.o_groups = gi(m, "attention.output_group_count", 0);
    c.o_rank = gi(m, "attention.output_lora_rank", 0);
    c.window = gi(m, "attention.sliding_window", 0);
    c.n_expert = m->n_expert;
    c.top_k = m->n_expert_used;
    c.n_ff = gi(m, "expert_feed_forward_length", 0);
    c.n_shared = m->n_expert_shared;
    c.route_scale = gf(m, "expert_weights_scale", 1.0);
    c.norm_topk = (int)gi(m, "expert_weights_norm", 1);
    {
        /* One clamp for every layer and arm; a per-layer schedule is refused. */
        const std::vector<int64_t> dummy;
        const char* s = gs(m, "swiglu_clamp_exp");
        c.swiglu_limit = s && *s && *s != '[' ? std::atof(s) : 10.0;
    }
    c.ih = gi(m, "attention.indexer.head_count", 0);
    c.idim = gi(m, "attention.indexer.key_length", 0);
    c.itopk = gi(m, "attention.indexer.top_k", 0);
    c.cand = gi(m, "attention.indexer.candidate_source_layer", -1);
    c.cand_blocks = gi(m, "attention.indexer.candidate_top_k_blocks", 0);
    c.cand_block = gi(m, "attention.indexer.candidate_block_size", 0);
    c.hc = gi(m, "hyper_connection.count", 0);
    c.hc_iters = gi(m, "hyper_connection.sinkhorn_iterations", 0);
    c.hc_eps = gf(m, "hyper_connection.epsilon", 0.0);
    c.eps = m->rms_eps;
    c.theta = m->rope_theta;
    c.comp_theta = gf(m, "attention.compress_rope_freq_base", 0.0);
    c.ratio = gl(m, "attention.compress_ratios");
    c.kv_src = gl(m, "attention.kv_source_layers");
    c.idx_src = gl(m, "attention.indexer.source_layers");
    c.ple = gl(m, "ple.layers");
    c.ngram = gi(m, "ple.ngram_size", 0);
    c.ple_heads = (c.ngram - 1) * gi(m, "ple.heads_per_ngram", 0);
    c.ple_dim = gi(m, "embedding_length_per_layer_input", 0);
    c.pad = gi(m, "engram.pad_token_id", -1);
    for (int64_t v : gl(m, "ple.head_vocab_sizes")) c.ple_rows += v;
    /* The release states each Engram table's rows; their sum is the joined table's. */
    if (!c.ple_rows)
        for (int64_t v : int_list(rad_meta_gets(m, "engram_num_embeddings", nullptr))) c.ple_rows += v;
    /* The compressed layers' rotary: YaRN with the bounds the reference derives from the
     * training length (model.py, precompute_freqs_cis). */
    c.yarn_factor = rad_meta_getf(m, "rope_scaling.factor", 0.0);
    const double orig = rad_meta_getf(m, "rope_scaling.original_max_position_embeddings", 0.0);
    const double bf = rad_meta_getf(m, "rope_scaling.beta_fast", 32.0);
    const double bs = rad_meta_getf(m, "rope_scaling.beta_slow", 1.0);
    if (c.yarn_factor > 1.0 && orig > 0 && c.comp_theta > 0 && c.rope > 0) {
        auto corr = [&](double r) {
            return (double)c.rope * std::log(orig / (r * 2.0 * std::numbers::pi)) / (2.0 * std::log(c.comp_theta));
        };
        c.yarn_low = std::max(std::floor(corr(bf)), 0.0);
        c.yarn_high = std::min(std::ceil(corr(bs)), (double)c.rope - 1);
    }

    const struct { const char* what; int64_t v; } need[] = {
        { "block_count", c.n_layer }, { "embedding_length", c.n_embd },
        { "attention.head_count", c.n_head }, { "attention.key_length", c.head_dim },
        { "rope.dimension_count", c.rope }, { "attention.q_lora_rank", c.q_lora },
        { "attention.output_group_count", c.o_groups }, { "attention.output_lora_rank", c.o_rank },
        { "attention.sliding_window", c.window }, { "expert_count", c.n_expert },
        { "expert_used_count", c.top_k }, { "expert_feed_forward_length", c.n_ff },
        { "hyper_connection.count", c.hc }, { "hyper_connection.sinkhorn_iterations", c.hc_iters },
        { "attention.indexer.head_count", c.ih }, { "attention.indexer.key_length", c.idim },
        { "attention.indexer.top_k", c.itopk } };
    for (const auto& n : need)
        if (n.v <= 0) {
            fprintf(stderr, "radiance: deepseek41: the checkpoint has no deepseek41.%s\n", n.what);
            return RAD_E_UNSUPPORTED;
        }
    /* The kv and index sources may be absent where no layer compresses (a model cut to its first
     * layers); a compressing layer without one is refused below. */
    if ((int64_t)c.ratio.size() < c.n_layer) {
        fprintf(stderr, "radiance: deepseek41: compress_ratios has %zu values for %lld layers\n",
                c.ratio.size(), (long long)c.n_layer);
        return RAD_E_UNSUPPORTED;
    }
    if (c.n_shared != 1 || c.head_dim % 2 || c.rope > c.head_dim || c.rope > c.idim ||
        (c.n_head * c.head_dim) % c.o_groups || c.n_expert % 2 || c.hc > 8) {
        fprintf(stderr, "radiance: deepseek41: unsupported geometry (shared experts %lld, head_dim "
                        "%lld, rope %lld, groups %lld, experts %lld, hc %lld)\n",
                (long long)c.n_shared, (long long)c.head_dim, (long long)c.rope,
                (long long)c.o_groups, (long long)c.n_expert, (long long)c.hc);
        return RAD_E_UNSUPPORTED;
    }
    /* Each compressing layer reads the latest kv source and index source at or below it; their
     * ratios must agree. */
    c.src_of.assign((size_t)c.n_layer, -1);
    c.idx_of.assign((size_t)c.n_layer, -1);
    int s = -1, x = -1;
    for (int l = 0; l < c.n_layer; ++l) {
        if (c.is_src(l)) s = l;
        if (c.is_idx(l)) x = l;
        const int64_t r = c.ratio[(size_t)l];
        if (r <= 0) continue;
        if (s < 0 || x < 0 || c.ratio[(size_t)s] != r || c.ratio[(size_t)x] != r ||
            c.head_dim % (2 * r) || c.idim % (2 * r)) {
            fprintf(stderr, "radiance: deepseek41: layer %d (ratio %lld) has no kv or index source "
                            "of its ratio\n", l, (long long)r);
            return RAD_E_UNSUPPORTED;
        }
        c.src_of[(size_t)l] = s;
        c.idx_of[(size_t)l] = x;
    }
    if (!c.ple.empty() && (c.ngram < 2 || c.ple_heads <= 0 || c.ple_dim <= 0 || c.pad < 0 ||
                           c.ple_rows <= 0)) {
        fprintf(stderr, "radiance: deepseek41: the Engram layout is incomplete\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the model */
struct Layer {
    int l = 0;
    int64_t r = 0;            /* compress ratio, 0 for window only */
    bool src = false, idx = false;
    /* weights */
    rad_weight hc_a_fn = 0, hc_a_scale = 0, hc_a_base = 0, hc_f_fn = 0, hc_f_scale = 0,
               hc_f_base = 0;
    rad_weight attn_norm = 0, wq_a = 0, q_norm = 0, wq_b = 0, wkv = 0, kv_norm = 0, sinks = 0,
               wo_b = 0;
    std::vector<rad_weight> wo_a;
    rad_weight comp_w = 0, comp_norm = 0, ik_w = 0, ik_norm = 0, iq_w = 0, iproj = 0;
    rad_weight e_mult = 0, e_vsz = 0, e_off = 0, e_kv = 0, e_gate = 0;
    /* ops, in issue order */
    rad_op e_ids = 0, e_emb = 0, e_copy = 0, e_gemm = 0, e_gate_op = 0, e_back = 0;
    /* This Engram layer's row ids [T, heads]: one buffer a layer, since step() writes every
     * layer's ids before the first block runs (see there). */
    rad_buf ids = 0;
    rad_op mix_a = 0, col_a = 0, norm_a = 0;
    rad_op q_a = 0, q_an = 0, q_b = 0, q_rope = 0;
    rad_op kv_g = 0, kv_n = 0, kv_rope = 0, kv_q = 0, kv_st = 0;
    rad_op c_gemm = 0, c_work = 0, c_pool = 0, c_norm = 0, c_tab = 0;
    rad_op k_gemm = 0, k_norm = 0, k_rope = 0, k_q = 0, k_st = 0;
    rad_op l_rope = 0, l_q = 0, l_st = 0, c_tail = 0;
    rad_op i_q = 0, i_rope = 0, i_qq = 0, i_w = 0, i_score = 0, i_cand = 0, i_topk = 0;
    /* index_score over run p of a step's rows (row_part p); i_runs[0] is i_score. */
    std::vector<rad_op> i_runs;
    rad_op attn = 0, o_rope = 0, o_q = 0, o_b = 0, post_a = 0, attn_fill = 0, ar_attn = 0;
    std::vector<rad_op> o_a;
    /* The fused chains (fused()), each 0 where its chain above is declared instead: kv_n + kv_rope
     * + kv_q, k_norm + k_rope + k_q, o_rope + o_q, l_rope + l_q, i_rope + i_qq, col_a + norm_a,
     * col_f + norm_f. */
    rad_op kv_f = 0, k_f = 0, o_f = 0, l_f = 0, i_f = 0, cn_a = 0, cn_f = 0;
    /* mhc_post into the next sublayer's collapse + norm (post_collapse_norm()): pcn_f = post_a +
     * col_f + norm_f, issued for post_a; pcn_a = the previous layer's post_f + this layer's col_a +
     * norm_a, issued for that post_f (so before this layer's Engram-free tap and mix_a). */
    rad_op pcn_f = 0, pcn_a = 0;
    rad_weight ffn_norm = 0;
    rad_op mix_f = 0, col_f = 0, norm_f = 0, post_f = 0;
    MoeFP8 moe{};
};

/* A dense linear's int8 activation pair, keyed by the GEMM op a linear() declared: the quantiser
 * writes the pair and the GEMM reads it. One pair for the model's widest activation keeps a
 * quantised trunk's arena small; a GEMM's own columns are sliced at issue. */
struct IqrQOp {
    rad_op  op = 0;
    int64_t K  = 0;
};

struct Model {
    Names nm{};
    Geom g{};
    Cfg c{};
    std::vector<Layer> layers;
    int64_t work = 0, cmax = 0;
    /* --calib-layers, hi exclusive. Both 0: the tap is off. */
    int calib_lo = 0, calib_hi = 0;
    /* The indexer scores a step's rows in runs of at most `irows` rows into one [irows, cmax] score
     * plane (index_runs()), at most `iparts` runs a step; `kwords` is index_candidates' keep, a bit
     * a block of the candidate layer's width, i32 words a row (0: the model has no candidates). */
    int64_t irows = 0, iparts = 0, kwords = 0;
    /* The attention world (geometry()): attention ranks [0, aw), this rank's index `ar` in it and
     * its `gr` output groups; `zero` for an attention-zero rank. */
    int aw = 1, ar = 0;
    int64_t gr = 0;
    bool zero = false;
    /* This rank's columns [sh_lo, sh_hi) of the shared expert when its width does not split
     * evenly into whole blocks (shared_slice); 0, 0 is the even split. */
    int64_t sh_lo = 0, sh_hi = 0;
    /* This rank's routed experts [ne_base, ne_base + ne_rank) of the model's numbering (rad_expert_split). */
    int64_t ne_rank = 0, ne_base = 0;
    /* buffers */
    rad_buf h0 = 0, h1 = 0, xcol = 0, xn = 0, pre_a = 0, pre_f = 0, post = 0, comb = 0;
    rad_buf qr = 0, qrn = 0, q = 0, kv = 0, kvn = 0, o = 0, oa = 0, attn_out = 0;
    rad_buf kvs = 0, stage = 0, page = 0, bpos = 0, nc = 0, pooled = 0, lat = 0, latn = 0,
            ik = 0, ikn = 0;
    rad_buf iq = 0, iw = 0, score = 0, keep = 0, sel = 0, cs_a = 0, cs_b = 0;
    rad_buf ctok = 0, emb = 0, embd = 0, ekv = 0;
    rad_buf final_hidden = 0, logit_input = 0, logits = 0;
    ActFP8 act{};
    MoeFP8::Wire mw{};
    /* KV groups */
    rad_kvgroup win = 0, tok = 0;
    std::map<int64_t, rad_kvgroup> comp, index, tail;
    /* --kv-cache-dtype fp8: the compressed latent and index-key entries as block-scaled E4M3
     * (rad_kv_fp8.h; kv_fmt()), the very values the bf16 cache holds. 0 keeps them bf16. */
    int lat_kvf = 0, idx_kvf = 0;
    /* The GGUF type of each linear weight the model ships in llama.cpp's blocks (lw()). */
    std::map<rad_weight, const char*> wty;
    /* IQR dense weights: lw()'s code view is `w`, its scale view `wsc`; `wdt` is the GEMM's dtype
     * (`i8a8h`, or an asymmetric rotated form) and `wrb` a trellis's rows per int8 scale. */
    std::map<rad_weight, rad_weight>   wsc;
    std::map<rad_weight, const char*>  wdt;
    std::map<rad_weight, int64_t>      wrb;
    /* The quantiser each dense GEMM waits for (linear()): the op and the K its pair was declared at. */
    std::map<rad_op, IqrQOp>           qops;
    rad_buf iqr_q8 = 0, iqr_s8 = 0;   /* [max_tok, widest K] i8; [max_tok, widest K / 128] f32 */
    int64_t win_bs = 0;
    /* model-wide weights and ops */
    rad_weight token = 0, head = 0, final_gain = 0, tmap = 0, ple_table = 0;
    /* The Engram table stored as affine planes (a GGUF's IQ4_NL): views gathered in place by
     * embed_lookup_planes, `ple_group` values a scale; 0 for a plain bf16 table. */
    rad_weight ple_codes = 0, ple_scale = 0, ple_tab = 0;
    int64_t ple_group = 0;
    rad_op embed = 0, expand = 0, pre0 = 0, pre1 = 0, rope_a = 0, rope_b_step = 0, tok_map = 0;
    rad_op contract = 0, fnorm = 0, gather = 0, head_op = 0, embed_ar = 0;
    rad_op cn_fin = 0;   /* contract + fnorm fused (fused()), or 0 */
    rad_op pcn_fin = 0;  /* the last post_f + contract + fnorm fused, or 0 */
};

static std::unique_ptr<Model> models[MAX_SLOTS];
static thread_local std::unique_ptr<Model> probe_model;

static rad_op op(RadBuilder* b, const char* name, std::vector<RadParam> ps,
                 std::vector<rad_weight> ws, std::initializer_list<rad_buf> rd,
                 std::initializer_list<rad_buf> wr) {
    const rad_op h = rad_decl_op(b, name, ps.data(), (int)ps.size(),
                                 ws.empty() ? nullptr : ws.data(), (int)ws.size());
    return rw(b, h, rd, wr);
}
static rad_weight wt(RadBuilder* b, const char* name, uint32_t dt,
                     std::initializer_list<int64_t> shape, RadWeightGroup grp,
                     int access = RAD_ACCESS_PER_TOKEN) {
    if (map_copy(b, name, name) != RAD_OK) return 0;
    return decl_w(b, name, dt, shape, access, RAD_SHARD_NONE, grp);
}
/* A weight split over the ranks: this rank's [lo, hi) of its split dimension (rows for
 * RAD_SHARD_ROW, columns for RAD_SHARD_COL), declared at `shape`, which must already be that
 * extent. An explicit span because the attention world can be smaller than the world. */
static rad_weight wts(RadBuilder* b, const Geom& g, const char* name, uint32_t dt,
                      std::initializer_list<int64_t> shape, RadWeightGroup grp, int shard, int64_t lo,
                      int64_t hi, int access = RAD_ACCESS_PER_TOKEN) {
    if (map_copy(b, name, name) != RAD_OK) return 0;
    const rad_weight w = decl_w(b, name, dt, shape, access, g.world > 1 ? shard : RAD_SHARD_NONE, grp);
    if (!w) {
        fprintf(stderr, "radiance: deepseek41: weight %s refused\n", name);
        return 0;
    }
    if (g.world > 1 && rad_weight_shard_span(b, w, lo, hi) != RAD_OK) {
        fprintf(stderr, "radiance: deepseek41: %s's span [%lld, %lld) refused\n", name, (long long)lo,
                (long long)hi);
        return 0;
    }
    return w;
}
/* A linear's weight, [n, k] at this rank's extents: llama.cpp's blocks when the model ships it
 * so (a GGUF weight, imported as its type's planes; rad_ggml.h), declared at the codes' dtype and
 * read by gemm_nt_ggml, else bf16. A block's values run along k, so k and a column span must cut
 * whole blocks. With `shard` and a world above one, this rank holds [lo, hi) of the split
 * dimension. */
static rad_weight lw(RadBuilder* b, Model& m, const char* name, int64_t n, int64_t k,
                     RadWeightGroup grp, int shard = RAD_SHARD_NONE, int64_t lo = 0, int64_t hi = 0,
                     bool mapped = false, int access = RAD_ACCESS_PER_TOKEN) {
    if (!mapped && map_copy(b, name, name) != RAD_OK) return 0;
    RadEncoding e{};
    int64_t blk = 0;
    const bool known = weight_enc(b, name, &e);
    const char* ty = known ? ggml_type_of(e, &blk) : nullptr;
    const bool split = m.g.world > 1 && shard != RAD_SHARD_NONE;
    const W8hForm form = known ? linear_w8h_form(e) : W8hForm::None;
    const bool w8 = known && linear_is_w8(e);
    const bool wnh = known && linear_is_wnh(e);
    /* A rotated or int8 narrow weight: lw()'s codes are a view of the stored planes, its scale
     * plane a second view; the GEMM decodes them where it reads them (libr4d's IQR rows). */
    if (form != W8hForm::None || w8 || wnh) {
        const bool trellis = form == W8hForm::Trellis;
        const RadEncPlane* tpl = trellis ? rad_enc_plane(&e, "trellis") : nullptr;
        const int64_t wb = trellis && tpl ? tpl->block[1] : 0;
        if ((n % 16) || (k % (trellis ? 256 : 128))) {
            fprintf(stderr, "radiance: deepseek41: %s is an IQR weight; a dense IQR GEMM takes rows in "
                            "16s and K in whole 256s, this rank reads %lldx%lld\n",
                    name, (long long)n, (long long)k);
            return 0;
        }
        if (split && shard == RAD_SHARD_COL && (lo % 128 || hi % 128)) {
            fprintf(stderr, "radiance: deepseek41: %s is an IQR weight, and this rank's column slice "
                            "[%lld, %lld) cuts a 128-column scale group\n",
                    name, (long long)lo, (long long)hi);
            return 0;
        }
        const char* cv = form == W8hForm::Zero    ? "codes,zero"
                       : form == W8hForm::Table ? "codes,table"
                       : form == W8hForm::Grid  ? "codes,grid"
                       : form == W8hForm::Trellis
                             ? (linear_w8h_mad(e) ? "codes,trellis,trellis.mad" : "codes,trellis")
                       : wnh                    ? "codes,zero"
                                                : "codes";
        std::string base(name);
        const bool suffix = base.size() > 6 && base.compare(base.size() - 6, 6, ".weight") == 0;
        if (suffix) base.resize(base.size() - 6);
        const char* sn = m.nm.f("%s.scale", base.c_str());
        const bool rowf = form != W8hForm::None && linear_w8h_rowf(e);
        const bool two = form != W8hForm::None && linear_w8h_scale2(e);
        const char* sv = wnh ? "scale,zero" : rowf ? "scale,scale.col" : two ? "scale,scale.1" : "scale";
        const uint32_t sdt = wnh ? (uint32_t)RAD_F16 : rowf ? (uint32_t)RAD_F32
                           : two ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16;
        const bool logical = wnh || rowf || two;
        const rad_weight w = decl_view(b, name, nullptr, cv, e.plane[0].dtype, { n, k }, access,
                                       split ? shard : RAD_SHARD_NONE, grp);
        const rad_weight ws = decl_view(b, sn, name, sv, sdt,
                                        { n, logical ? k : k / RAD_FP8_BLOCK }, access,
                                        split ? shard : RAD_SHARD_NONE, grp);
        if (!w || !ws) {
            fprintf(stderr, "radiance: deepseek41: IQR weight %s refused\n", name);
            return 0;
        }
        if (split && (rad_weight_shard_span(b, w, lo, hi) != RAD_OK ||
                      rad_weight_shard_span(b, ws, logical || shard == RAD_SHARD_ROW ? lo : lo / RAD_FP8_BLOCK,
                                            logical || shard == RAD_SHARD_ROW ? hi : hi / RAD_FP8_BLOCK) != RAD_OK)) {
            fprintf(stderr, "radiance: deepseek41: %s's IQR span [%lld, %lld) refused\n",
                    name, (long long)lo, (long long)hi);
            return 0;
        }
        m.wsc[w] = ws;
        m.wdt[w] = wnh ? (e.plane[0].dtype == RAD_U5 ? "w5a8i8" : "w4a8i8_asym")
                       : form != W8hForm::None ? "i8a8h" : "i8a8";
        m.wrb[w] = wb;
        return w;
    }
    const bool split_ggml = split;
    if (ty && (k % blk || (split_ggml && shard == RAD_SHARD_COL && (lo % blk || hi % blk)))) {
        fprintf(stderr, "radiance: deepseek41: %s is GGUF %s, whose blocks cover %lld of K; this rank "
                        "reads %lld columns%s\n", name, ty, (long long)blk, (long long)k,
                split_ggml && shard == RAD_SHARD_COL ? " of a slice" : "");
        return 0;
    }
    const rad_weight w = decl_w(b, name, ty ? ggml_codes_dtype(e) : (uint32_t)RAD_BF16, { n, k },
                                access, split_ggml ? shard : RAD_SHARD_NONE, grp);
    if (!w) {
        fprintf(stderr, "radiance: deepseek41: weight %s refused\n", name);
        return 0;
    }
    if (split_ggml && rad_weight_shard_span(b, w, lo, hi) != RAD_OK) {
        fprintf(stderr, "radiance: deepseek41: %s's span [%lld, %lld) refused\n", name, (long long)lo,
                (long long)hi);
        return 0;
    }
    if (ty) m.wty[w] = ty;
    return w;
}
/* y = x . w^T over lw()'s weight: for a GGUF block it is gemm_nt_ggml, for an IQR view gemm_nt_q
 * at the weight's code dtype over a rotated int8 activation pair, otherwise gemm_nt at bf16. */
static rad_op linear(RadBuilder* b, Model& m, int64_t rows, rad_weight w, int64_t n,
                     int64_t k, rad_buf x, rad_buf y) {
    const auto s = m.wsc.find(w);
    if (s != m.wsc.end()) {
        const char* qdt = m.wdt.at(w);
        const bool rot = std::strcmp(qdt, "i8a8") != 0;
        const rad_op qop = op(b, rot ? "had_quant_act_i8g" : "quant_act_i8g",
                              { RAD_RANGE("M", 1, rows), RAD_INT("n", k), RAD_INT("group", RAD_FP8_BLOCK),
                                RAD_STR("dtype", "bf16") },
                              {}, { x }, { m.iqr_q8, m.iqr_s8 });
        std::vector<RadParam> ps = { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k),
                                     RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", qdt) };
        const int64_t wrb = m.wrb.at(w);
        if (wrb > 1) ps.push_back(RAD_INT("w_rb", wrb));
        std::vector<rad_weight> wts = { w, s->second };
        const rad_op h = op(b, "gemm_nt_q", ps, wts, { m.iqr_q8, m.iqr_s8 }, { y });
        if (qop && h) m.qops[h] = IqrQOp{ qop, k };
        return h;
    }
    const auto t = m.wty.find(w);
    if (t != m.wty.end())
        return op(b, "gemm_nt_ggml",
                  { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("type", t->second),
                    RAD_STR("dtype", m.g.dtype) },
                  { w }, { x }, { y });
    return op(b, "gemm_nt",
              { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("dtype", m.g.dtype) },
              { w }, { x }, { y });
}
/* Issue a dense linear: its quantiser first when lw() found an IQR weight, then its GEMM. The pair
 * is one buffer for the whole model, so the quantiser and GEMM read the same columns of it. */
static void lin_issue(RadCtx* x, Model& m, rad_op h, int64_t rows, RadOperand a, rad_weight w,
                      RadOperand y) {
    const auto q = m.qops.find(h);
    if (q != m.qops.end()) {
        const int64_t K = q->second.K;
        RAD_ISSUE_N(x, q->second.op, rows, a, bcol(m.iqr_q8, 0, K, rows),
                    bcol(m.iqr_s8, 0, K / RAD_FP8_BLOCK, rows));
    }
    RAD_ISSUE_N(x, h, rows, a, RAD_W(w), y);
}

static rad_op rmsnorm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n,
                      rad_buf x, rad_buf y) {
    return op(b, "rmsnorm",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", m.c.eps),
                RAD_STR("dtype", "bf16") },
              { w }, { x }, { y });
}
static rad_op rope_rows(RadBuilder* b, const Model& m, int64_t rows, int64_t heads, int64_t hd,
                        int inverse, rad_buf x, rad_buf cs) {
    return op(b, "ds41_rope_rows",
              { RAD_RANGE("M", 1, rows), RAD_INT("n_head", heads), RAD_INT("head_dim", hd),
                RAD_INT("rot", m.c.rope), RAD_INT("off", hd - m.c.rope), RAD_STR("mode", "gptj"),
                RAD_INT("inverse", inverse), RAD_STR("dtype", "bf16") },
              {}, { x, cs }, { x });
}
static rad_op roundtrip(RadBuilder* b, int64_t rows, int64_t n, int64_t block, const char* fmt,
                        const char* scale, rad_buf x) {
    return op(b, "quant_roundtrip",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("block", block),
                RAD_STR("format", fmt), RAD_STR("scale", scale), RAD_STR("dtype", "bf16") },
              {}, { x }, { x });
}
static rad_op entry_store(RadBuilder* b, int64_t rows, int64_t n, int64_t bs, rad_buf x,
                          rad_buf page, const char* format) {
    std::vector<RadParam> ps = { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("block_size", bs),
                                 RAD_STR("dtype", "bf16") };
    if (format) ps.push_back(RAD_STR("format", format));
    return op(b, "entry_store", ps, {}, { x, page }, {});
}

/* ---- fused chains (docs/OPS.md rmsnorm_rope_roundtrip, rope_rows_roundtrip,
 * mhc_collapse_rmsnorm): one launch for a chain of two or three dependent ops on the same rows,
 * the same bytes as the chain (a kernel library's `replaces`). A decode step pays a fixed cost per
 * dependent launch, so this is most of what the small ops cost there. They are device-only (libr4d
 * declares them; libref has no row): each declares the fused op only where a loaded library
 * declares and serves it, and returns 0 otherwise, and the caller then declares the chain. */
static rad_op fused(RadBuilder* b, const char* name, std::vector<RadParam> ps,
                    std::vector<rad_weight> ws, std::initializer_list<rad_buf> rd,
                    std::initializer_list<rad_buf> wr) {
    if (!rad_op_declared(b, name)) return 0;
    const rad_op h = rad_decl_op(b, name, ps.data(), (int)ps.size(),
                                 ws.empty() ? nullptr : ws.data(), (int)ws.size());
    if (!h || !rad_op_resolved(b, h)) return 0;
    return rw(b, h, rd, wr);
}
/* rmsnorm (x, w -> y) then rope_rows and roundtrip over y, as one op: one head of n columns. */
static rad_op norm_rope_rt(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n,
                           int64_t block, const char* fmt, const char* scale, rad_buf x, rad_buf y,
                           rad_buf cs) {
    return fused(b, "rmsnorm_rope_roundtrip",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", m.c.eps),
                   RAD_INT("n_head", 1), RAD_INT("head_dim", n), RAD_INT("rot", m.c.rope),
                   RAD_INT("off", n - m.c.rope), RAD_STR("mode", "gptj"), RAD_INT("inverse", 0),
                   RAD_INT("block", block), RAD_STR("format", fmt), RAD_STR("scale", scale),
                   RAD_STR("dtype", "bf16") },
                 { w }, { x, cs }, { y });
}
/* rope_rows then roundtrip over all heads * hd columns of x, in place, as one op. */
static rad_op rope_rt(RadBuilder* b, const Model& m, int64_t rows, int64_t heads, int64_t hd,
                      int inverse, int64_t block, const char* fmt, const char* scale, rad_buf x,
                      rad_buf cs) {
    return fused(b, "rope_rows_roundtrip",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n_head", heads), RAD_INT("head_dim", hd),
                   RAD_INT("rot", m.c.rope), RAD_INT("off", hd - m.c.rope), RAD_STR("mode", "gptj"),
                   RAD_INT("inverse", inverse), RAD_INT("block", block), RAD_STR("format", fmt),
                   RAD_STR("scale", scale), RAD_STR("dtype", "bf16") },
                 {}, { x, cs }, { x });
}
/* mhc_collapse (h, pre -> x) then rmsnorm (x, w -> y), as one op. */
static rad_op collapse_norm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, rad_buf h,
                            rad_buf pre, rad_buf x, rad_buf y) {
    return fused(b, "mhc_collapse_rmsnorm",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", m.c.n_embd), RAD_INT("hc", m.c.hc),
                   RAD_F64("eps", m.c.eps), RAD_STR("dtype", "bf16") },
                 { w }, { h, pre }, { x, y });
}
/* mhc_post (in, h, post, comb -> out), mhc_collapse (out, pre -> x), rmsnorm (x, w -> y), as one
 * op. */
static rad_op post_collapse_norm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w,
                                 rad_buf in, rad_buf h, rad_buf out, rad_buf pre, rad_buf x,
                                 rad_buf y) {
    return fused(b, "mhc_post_collapse_rmsnorm",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", m.c.n_embd), RAD_INT("hc", m.c.hc),
                   RAD_F64("eps", m.c.eps), RAD_STR("dtype", "bf16") },
                 { w }, { in, h, m.post, m.comb, pre }, { out, x, y });
}

/* Row cap of libr4d's ar_mhc_post_collapse_rmsnorm. Declaration and the row the MoE block hands
 * off share this bound, so a prefill wider than it keeps the separate all-reduce. */
/* The shared expert is split by width, and a rank's K slice of its down projection must fall on
 * the encoding's block boundary along K: 256 columns for the GGUF k- and i-quant superblocks, 128 for
 * block fp8, 32 for legacy GGUF and MXFP4, 1 for bf16. A slice on a 256-column boundary is on every
 * one's. The even split is kept wherever each rank's share is whole 256-column blocks (and wherever
 * the width is no whole number of them, where the encoding decides at declare, as before); otherwise
 * the width splits by whole blocks, the first nb % world ranks one block more. DeepSeek-V4.1-Flash's
 * 2304 columns are nine blocks: three ranks take three each, and four ranks 768/512/512/512 where
 * the even 576 would cut a superblock. Each rank computes its gate/up columns and K slice of down,
 * and the all-reduce sums the partials (MoeFP8::Config::shared_lo). */
static constexpr int64_t kSharedBlock = 256;

static void shared_slice(int64_t n_ff, int world, int rank, int64_t* lo, int64_t* hi) {
    *lo = *hi = 0;
    if (world <= 1 || n_ff % kSharedBlock) return;
    if (n_ff % world == 0 && (n_ff / world) % kSharedBlock == 0) return;
    const int64_t nb = n_ff / kSharedBlock, base = nb / world, extra = nb % world;
    const int64_t b0 = rank * base + std::min<int64_t>(rank, extra);
    *lo = b0 * kSharedBlock;
    *hi = (b0 + base + (rank < extra ? 1 : 0)) * kSharedBlock;
}

static int geometry(Model& m, const RadBuildCtx* ctx) {
    auto& g = m.g;
    const auto& c = m.c;
    g.n_layer = c.n_layer;
    g.n_embd = c.n_embd;
    g.head_dim = c.head_dim;
    const int world = ctx->world_size ? ctx->world_size : 1;
    g.world = world;
    g.rank = ctx->rank;
    /* The attention world: the largest d <= world that divides both the output groups and the
     * heads; the ranks past it are attention-zero (Geom::attn_world). There is one latent KV head,
     * so geom_from's KV-head rule does not apply: three ranks attend on two, four on four. */
    int aw = 1;
    for (int d = world; d >= 1; --d)
        if (c.o_groups % d == 0 && c.n_head % d == 0) {
            aw = d;
            break;
        }
    g.attn_world = aw;
    g.attn_zero = ctx->rank >= aw;
    g.attn_rank = g.attn_zero ? aw - 1 : ctx->rank;
    m.aw = aw;
    m.ar = g.attn_rank;
    m.zero = g.attn_zero;
    m.gr = c.o_groups / aw;
    g.n_head_all = c.n_head;
    g.n_head = c.n_head / aw;
    g.n_head_kv = g.n_head_kv_all = 1;
    /* Routed experts are expert-parallel: each rank holds n_expert / world whole experts, an even
     * count (the grouped GEMM takes two parity tables), and the router stays replicated. Whole
     * experts are what the bf16 and quantised expert paths both serve, at any world (an expert's
     * width split into whole quant blocks holds at three ranks but not at four). The shared expert
     * is split by width. */
    int64_t sh_lo = 0, sh_hi = 0;
    shared_slice(c.n_ff, world, ctx->rank, &sh_lo, &sh_hi);
    /* The routed experts a rank holds: n_expert / world each, or RADIANCE_EXPERT_SPLIT's list (the
     * plugin's own RADIANCE_DS41_EXPERT_SPLIT is still read); rad_expert_split.h. Even counts: the
     * grouped GEMM takes two tables by parity. */
    bool split_given = false;
    const char* split_env = "";
    if (!rad_expert_split(c.n_expert, world, ctx->rank, 2, "RADIANCE_DS41_EXPERT_SPLIT", &m.ne_rank, &m.ne_base,
                          &split_given, &split_env)) {
        if (split_given)
            fprintf(stderr, "radiance: deepseek41: %s=%s does not split %lld experts over "
                            "%d ranks (one even count a rank, comma-separated, summing to the model's)\n",
                    split_env, std::getenv(split_env), (long long)c.n_expert, world);
        else
            fprintf(stderr, "radiance: deepseek41: %lld experts do not split over %d ranks (an even count of "
                            "whole experts each)\n", (long long)c.n_expert, world);
        return RAD_E_UNSUPPORTED;
    }
    if (sh_hi == 0 && c.n_ff % world) {
        fprintf(stderr, "radiance: deepseek41: the shared expert's width %lld does not split over %d ranks\n",
                (long long)c.n_ff, world);
        return RAD_E_UNSUPPORTED;
    }
    g.n_ff_all = c.n_ff;
    /* This rank's width of the shared expert (its buffers and the MoE block's n_ff_shared). */
    g.n_ff = sh_hi ? sh_hi - sh_lo : c.n_ff / world;
    m.sh_lo = sh_lo;
    m.sh_hi = sh_hi;
    g.n_vocab_all = c.n_vocab;
    g.n_vocab = shard_extent(c.n_vocab, world, ctx->rank);
    g.vocab_off = shard_begin(c.n_vocab, world, ctx->rank);
    g.max_tok = ctx->max_tok;
    g.max_ctx = ctx->max_ctx;
    g.max_seqs = ctx->max_seqs;
    g.max_spec = ctx->max_spec;
    g.max_logit_rows = std::min<int64_t>(g.max_tok, g.max_seqs * (g.max_spec + 1));
    if (ctx->max_out_rows > g.max_logit_rows) g.max_logit_rows = ctx->max_out_rows;
    g.eps = (float)c.eps;
    g.theta = (float)c.theta;
    g.dtype = "bf16";
    g.act_dtype = RAD_BF16;
    g.w_bf16 = true;
    /* The trunk mixes bf16 and quantised linears: each quantised one takes its own form. */
    g.w_bf16_mixed = true;
    /* The all-reduce wire is exact unless --tp-wire wht6 asks for the lossy one (local, 2026-11-11: the
     * branch pinned it exact; libr4d has the tiered-int8 two-shot at three and four ranks and the
     * rotated 6-bit at two, for bf16 messages from --tp-wire-min-kb up, which here are the prefill
     * chunks'). A world or a message libr4d has no lossy row for stops the start by name. */
    g.wire_exact = ctx->tp_wire_lossy ? 0 : 1;
    g.wire_min_bytes = ctx->tp_wire_min_bytes;
    return g.max_tok < 1 || g.max_ctx < 1 || g.max_seqs < 1 ? RAD_E_INVAL : RAD_OK;
}

/* index_candidates' keep words a row (docs/OPS.md): a bit a block of `block` positions. */
static int64_t keep_words(int64_t width, int64_t block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

/* The indexer's score plane is a row's score of every compressed position, [rows, cmax] f32 with
 * cmax = max_ctx / (the smallest ratio) + 1: at ratio 1, 8192-row chunks and 256k of context a
 * whole step's plane would be 8 GiB, and the candidates' mask (live from the candidate layer to the
 * last that reads it) as much again. So a step is scored in runs of rows into one plane of at most
 * kScorePlane bytes (but never fewer than kScoreRunMin rows), each run selected from before the next
 * overwrites it, and the mask is a bit a block (index_candidates).
 *
 * A step of T rows takes n = ceil(T / irows) runs of R = ceil(T / n) rows. Rows are independent,
 * so every run is the unsplit op's rows bit for bit, provided each run is scored by the kernel the
 * whole step would have taken: kernels pick their form by the pass height (libr4d's index_score
 * runs WMMA above 48 rows, a different dot order from the decode form's), and a run of a split
 * step is over irows / 2 >= 128 rows, as tall as any form's threshold. A step of at most irows
 * rows is one run, the op as before. */
static constexpr int64_t kScorePlane = 256ll << 20;
static constexpr int64_t kScoreRunMin = 256;
static void index_runs(Model& m) {
    const auto& g = m.g;
    const auto& c = m.c;
    m.irows = std::min(g.max_tok, std::max(kScoreRunMin, kScorePlane / (4 * m.cmax)));
    m.iparts = (g.max_tok + m.irows - 1) / m.irows;
    m.kwords = 0;
    if (c.cand >= 0 && c.cand < c.n_layer && c.cand_block > 0 && c.ratio[(size_t)c.cand] > 0)
        m.kwords = keep_words(g.max_ctx / c.ratio[(size_t)c.cand] + 1, c.cand_block);
}

static int buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int64_t T = g.max_tok, D = c.n_embd, W = m.work;
    const int64_t HN = c.hc * D, HD = g.n_head * c.head_dim;
    auto bf = [&](const char* n, std::initializer_list<int64_t> s) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, s);
    };
    m.h0 = bf("streams", { T, HN });
    m.h1 = bf("streams_next", { T, HN });
    m.xcol = bf("collapsed", { T, D });
    m.xn = bf("normed", { T, D });
    m.pre_a = decl_b(b, nm.f("pre_attn"), RAD_F32, { T, c.hc });
    m.pre_f = decl_b(b, nm.f("pre_ffn"), RAD_F32, { T, c.hc });
    m.post = decl_b(b, nm.f("post"), RAD_F32, { T, c.hc });
    m.comb = decl_b(b, nm.f("comb"), RAD_F32, { T, c.hc * c.hc });
    m.qr = bf("q_lora", { T, c.q_lora });
    m.qrn = bf("q_lora_normed", { T, c.q_lora });
    m.q = bf("q", { T, HD });
    m.kv = bf("window_kv", { T, c.head_dim });
    m.kvn = bf("window_kv_normed", { T, c.head_dim });
    m.o = bf("attn_o", { T, HD });
    m.oa = bf("attn_o_lora", { T, m.gr * c.o_rank });
    m.attn_out = bf("attn_out", { T, D });
    m.kvs = decl_b(b, nm.f("comp_kv_score"), RAD_F32, { T, 2 * c.head_dim });
    m.stage = decl_b(b, nm.f("comp_stage"), RAD_F32, { 2 * W, 2 * c.head_dim });
    m.page = decl_b(b, nm.f("comp_page"), RAD_I32, { W });
    m.bpos = decl_b(b, nm.f("comp_bpos"), RAD_I32, { W });
    m.nc = decl_b(b, nm.f("comp_nc"), RAD_I32, { g.max_seqs });
    const int64_t LR = std::max(T, W);
    m.pooled = bf("comp_pooled", { LR, c.head_dim });
    m.lat = bf("comp_latent", { LR, c.head_dim });
    m.latn = bf("comp_latent_normed", { LR, c.head_dim });
    m.ik = bf("index_k", { LR, c.idim });
    m.ikn = bf("index_k_normed", { LR, c.idim });
    m.iq = bf("index_q", { T, c.ih * c.idim });
    m.iw = bf("index_w", { T, c.ih });
    m.score = decl_b(b, nm.f("index_score"), RAD_F32, { m.irows, m.cmax });
    if (m.kwords > 0) m.keep = decl_b(b, nm.f("index_keep"), RAD_I32, { T, m.kwords });
    m.sel = decl_b(b, nm.f("index_sel"), RAD_I32, { T, c.itopk });
    m.cs_a = decl_b(b, nm.f("rope_window"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    m.cs_b = decl_b(b, nm.f("rope_compressed"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    if (!c.ple.empty()) {
        m.ctok = decl_b(b, nm.f("engram_tokens"), RAD_I32, { T });
        /* The gathered rows are host domain, as rad_block_ple.h's: a host gather (the table on the
         * disk or RAM tier) whose operands are all host-arena buffers joins on the last device op
         * that touched the host arena, where a device-domain operand drains the stream
         * (core/runtime/issue.cpp). A cast copies them onto the card for the GEMM, which reads
         * each row band once per column block. */
        m.emb = decl_b(b, nm.f("engram_rows"), RAD_BF16, { T, c.ple_heads * c.ple_dim },
                       RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
        m.embd = bf("engram_rows_dev", { T, c.ple_heads * c.ple_dim });
        m.ekv = bf("engram_kv", { T, (c.hc + 1) * D });
    }
    /* The routed FFN's wires (rad_block_moe_fp8.h), bf16 throughout. */
    /* bf16 linears: each ActFP8 carries only its activation (`x`); the plugin norms the FFN input
     * itself (MoeFP8's external-input form) and the block leaves its delta in `act.x`. */
    m.act.x = bf("ffn_normed", { T, D });
    /* A quantised dense trunk reads this pair, one f32 scale a 128 columns, shared by every dense
     * IQR GEMM. The activation quantiser (had_quant_act_i8g) checks the code and scale planes
     * against the linear's own K, and a row slice does not narrow the column, so the pair is the
     * hidden width -- the width every dense linear but the Engram gather reads. An Engram linear
     * wider than that declares its own pair. */
    const int64_t QK = D;
    m.iqr_q8 = decl_b(b, nm.f("iqr_q8"), RAD_I8, { T, QK });
    m.iqr_s8 = decl_b(b, nm.f("iqr_s8"), RAD_F32, { T, fp8_blocks(QK) });
    /* The shared expert's input is the same normed activation; its own quantiser writes the pair. */
    m.act.q8 = m.iqr_q8;
    m.act.s8 = m.iqr_s8;
    auto& mw = m.mw;
    mw.x = m.xcol;
    mw.h = m.act;
    mw.logits = decl_b(b, nm.f("router_logits"), RAD_F32, { T, c.n_expert });
    mw.ids = decl_b(b, nm.f("expert_ids"), RAD_I32, { T, c.top_k });
    mw.ew = decl_b(b, nm.f("expert_weights"), RAD_F32, { T, c.top_k });
    mw.eoff = decl_b(b, nm.f("expert_offsets"), RAD_I32, { c.n_expert + 1 });
    /* Persistent, as glm5next's: the file stager reads a layer's counts at its trigger from the
     * pointer the last pass reported (Ctx::route_counts_device); a transient buffer's bytes may
     * belong to another buffer by then, so the stager would read every routed layer whole. */
    mw.ecnt = decl_b(b, nm.f("expert_counts"), RAD_I32, { c.n_expert }, RAD_BUF_PERSIST);
    mw.sorted = decl_b(b, nm.f("expert_sorted"), RAD_I32, { T * c.top_k });
    mw.egu = bf("expert_gate_up", { T * c.top_k, 2 * c.n_ff });
    mw.eff.x = bf("expert_ff", { T * c.top_k, c.n_ff });
    mw.edn = bf("expert_down", { T * c.top_k, D });
    /* the shared expert: this rank's width; none where its slice is empty (shared_slice) */
    if (g.n_ff > 0) {
        mw.sgu = bf("shared_gate_up", { T, 2 * g.n_ff });
        mw.sff.x = bf("shared_ff", { T, g.n_ff });
    }
    mw.sout = bf("shared_out", { T, D });
    mw.sgate = bf("shared_one", { T, 1 });
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, D }, RAD_BUF_PERSIST);
    m.logit_input = bf("logit_input", { g.max_logit_rows, D });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    return rad_declare_logits(b, m.logits);
}

/* The caches. The window holds the whole rotated latent per slot (K then V halves); a compressed or
 * index group holds one entry a block of `ratio` slots (entry_store fixes the block size); the
 * compressor's tail is a per-sequence ring of r + max_spec f32 rows. Declared identically on every
 * rank (an attention-zero rank zeroes them). */
static rad_kvgroup win_group(RadBuilder* b, Model& m) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_WINDOW;
    d.dtype = RAD_BF16;
    d.window = m.c.window + m.g.max_spec;
    d.n_head_kv = 1;
    d.head_dim = m.c.head_dim / 2;
    return rad_decl_kv_group(b, m.nm.f("kv_window"), &d);
}
static rad_kvgroup tail_group(RadBuilder* b, Model& m, int64_t r) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_F32;
    d.n_head_kv = 1;
    d.state_dim[0] = r + m.g.max_spec;
    d.state_dim[1] = 2 * m.c.head_dim;
    return rad_decl_kv_group(b, m.nm.f("kv_comp_tail_r%lld", (long long)r), &d);
}
/* An entry format `kvf` (rad_kv_fp8.h) for a group of n values one entry a block of r slots: kvf
 * when its bytes fill the r slots' K and V halves evenly, else 0 (bf16). */
static int kv_fmt(int kvf, int64_t n, int64_t r) {
    if (kvf <= 0 || r < 1) return 0;
    const int64_t e = rad_kvf_bytes(kvf, n);
    return e % (2 * r) == 0 ? kvf : 0;
}
static const char* kv_fmt_name(int kvf) {
    return kvf == RAD_KVF_E4M3_E4M3_16 ? "e4m3_e4m3_16" : (kvf == RAD_KVF_E4M3_UE8M0_32 ? "e4m3_ue8m0_32" : nullptr);
}
/* A compressed or index group: bf16 rows of n values, or (kvf) E4M3 rows of the entry's bytes. */
static rad_kvgroup full_group(RadBuilder* b, Model& m, const char* fmt, int64_t n, int64_t r, int kvf) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_FULL;
    d.dtype = kvf ? RAD_F8E4M3 : RAD_BF16;
    d.n_head_kv = 1;
    d.head_dim = (kvf ? rad_kvf_bytes(kvf, n) : n) / r / 2;
    return rad_decl_kv_group(b, m.nm.f(fmt, (long long)r), &d);
}

/* The routed FFN's checkpoint tensors at (absolute) layer `l`: a GGUF's stacks (ffn_gate_exps,
 * ffn_up_exps, ffn_down_exps, each [n_expert, ...]), or the release's experts one by one
 * (`<layer>.ffn.experts.<e>.w1/w3/w2`, gate/up/down as model.py's Expert names them). */
static void moe_names(Model& m, int l, MoeFP8::Src* s) {
    auto& nm = m.nm;
    s->router = nm.f("blk.%d.ffn_gate_inp.weight", l);
    s->router_bias = nm.f("blk.%d.exp_probs_b.bias", l);
    s->shared_mlp_gate = nm.f("blk.%d.ffn_gate_shexp.weight", l);
    s->shared_mlp_up = nm.f("blk.%d.ffn_up_shexp.weight", l);
    s->shared_mlp_down = nm.f("blk.%d.ffn_down_shexp.weight", l);
    s->experts_gate = nm.f("blk.%d.ffn_gate_exps.weight", l);
    s->experts_up = nm.f("blk.%d.ffn_up_exps.weight", l);
    s->experts_down = nm.f("blk.%d.ffn_down_exps.weight", l);
}

/* ------------------------------------------------------------------ one layer */
static int layer(RadBuilder* b, Model& m, Layer& L, int l) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int64_t T = g.max_tok, D = c.n_embd, HD = g.n_head * c.head_dim, W = m.work;
    const int64_t hd = c.head_dim, mix = c.hc * (c.hc + 2);
    const auto grp = grp_layer(l);
    L.l = l;
    L.r = c.ratio[(size_t)l];
    L.src = c.is_src(l);
    L.idx = c.is_idx(l);
    auto w = [&](const char* fmt, uint32_t dt, std::initializer_list<int64_t> shape) {
        return wt(b, nm.f(fmt, l), dt, shape, grp);
    };

    /* ---- Engram, on the streams in front of the block */
    const int pi = c.ple_index(l);
    if (pi >= 0) {
        const int64_t H = c.ple_heads;
        L.e_mult = decl_w(b, nm.f("blk.%d.engram_mult", l), RAD_I64, { c.ngram },
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        L.e_vsz = decl_w(b, nm.f("blk.%d.engram_vsz", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_off = decl_w(b, nm.f("blk.%d.engram_off", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_kv = lw(b, m, nm.f("blk.%d.engram_kv.weight", l), (c.hc + 1) * D, H * c.ple_dim, grp);
        L.e_gate = w("blk.%d.engram_gate.weight", RAD_F32, { c.hc * D });
        if (!m.tok) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_CONV;
            d.dtype = RAD_I32;
            d.conv_width = c.ngram;
            d.n_head_kv = 1;
            d.head_dim = 1;
            m.tok = rad_decl_kv_group(b, nm.f("kv_engram_tokens"), &d);
            if (!m.tok) return RAD_E_INVAL;
        }
        DS_TRY(rad_bind_layer_kv(b, l, m.tok));
        /* Host domain, like the rows (see m.emb), and persistent rather than liveness-shared:
         * liveness runs over the declared op order, and step() writes these ids before the first
         * block, far ahead of this layer's gather. */
        L.ids = decl_b(b, nm.f("blk.%d.engram_ids", l), RAD_I32, { T, H }, RAD_BUF_PERSIST,
                       RAD_DOMAIN_HOST);
        L.e_ids = op(b, "ngram_ids",
                     { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("ngram", c.ngram),
                       RAD_INT("eos", c.pad) },
                     { L.e_mult, L.e_vsz, L.e_off }, { m.ctok }, { L.ids });
        if (m.ple_group > 0 && m.ple_tab)
            L.e_emb = op(b, "embed_lookup_planes",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", c.ple_rows), RAD_INT("group", m.ple_group),
                           RAD_STR("dtype", "bf16") },
                         { m.ple_codes, m.ple_scale, m.ple_tab }, { L.ids }, { m.emb });
        else if (m.ple_group > 0)
            L.e_emb = op(b, "embed_lookup_planes",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", c.ple_rows), RAD_INT("group", m.ple_group),
                           RAD_STR("dtype", "bf16") },
                         { m.ple_codes, m.ple_scale }, { L.ids }, { m.emb });
        else
            L.e_emb = op(b, "embed_lookup",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", c.ple_rows), RAD_STR("dtype", "bf16") },
                         { m.ple_table }, { L.ids }, { m.emb });
        L.e_copy = op(b, "cast",
                      { RAD_RANGE("M", 1, T), RAD_INT("n", H * c.ple_dim), RAD_STR("from", "bf16"),
                        RAD_STR("to", "bf16") },
                      {}, { m.emb }, { m.embd });
        L.e_gemm = linear(b, m, T, L.e_kv, (c.hc + 1) * D, H * c.ple_dim, m.embd, m.ekv);
        L.e_gate_op = op(b, "stream_gate_add",
                         { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                           RAD_F64("eps", c.eps), RAD_F64("gate_eps", 1e-6),
                           RAD_STR("dtype", "bf16") },
                         { L.e_gate }, { m.h0, m.ekv }, { m.h1 });
        /* The block reads the streams from h0: the gated streams go back there. */
        L.e_back = op(b, "cast",
                      { RAD_RANGE("M", 1, T), RAD_INT("n", c.hc * D), RAD_STR("from", "bf16"),
                        RAD_STR("to", "bf16") },
                      {}, { m.h1 }, { m.h0 });
    }

    /* ---- the attention connection */
    L.hc_a_fn = w("blk.%d.hc_attn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_a_scale = w("blk.%d.hc_attn_scale.weight", RAD_F32, { 3 });
    L.hc_a_base = w("blk.%d.hc_attn_base.weight", RAD_F32, { mix });
    const std::vector<RadParam> mixp = { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                         RAD_INT("iterations", c.hc_iters), RAD_F64("eps", c.eps),
                                         RAD_F64("hc_eps", c.hc_eps), RAD_STR("dtype", "bf16") };
    const std::vector<RadParam> hcp = { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                        RAD_STR("dtype", "bf16") };
    L.mix_a = op(b, "mhc_mix", mixp, { L.hc_a_fn, L.hc_a_scale, L.hc_a_base }, { m.h0 },
                 { m.pre_a, m.post, m.comb });
    if (!m.zero) {
        L.attn_norm = w("blk.%d.attn_norm.weight", RAD_F32, { D });
        /* The previous layer's post_f writes h0 straight into this collapse unless an Engram
         * block (above) rewrites h0 between them. */
        if (l > 0 && !L.e_ids)
            L.pcn_a = post_collapse_norm(b, m, T, L.attn_norm, m.act.x, m.h1, m.h0, m.pre_f,
                                         m.xcol, m.xn);
        if (!L.pcn_a) L.cn_a = collapse_norm(b, m, T, L.attn_norm, m.h0, m.pre_f, m.xcol, m.xn);
    }
    if (!L.cn_a && !L.pcn_a) L.col_a = op(b, "mhc_collapse", hcp, {}, { m.h0, m.pre_f }, { m.xcol });
    if (m.zero) {
        /* Attention-zero (Geom::attn_world): no heads on this rank. Its caches are declared and
         * bound as on an attention rank, the weightless stores that fix their block sizes declared
         * but never issued, then zeroed (rad_kv_group_zero); the block's output is a fill of zeros
         * the all-reduce adds. */
        if (!m.win && !(m.win = win_group(b, m))) return RAD_E_INVAL;
        L.kv_st = op(b, "kv_store",
                     { RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                       RAD_INT("block_size", 16), RAD_STR("kv_dtype", "bf16") },
                     {}, { m.kvn }, {});
        if (!m.win_bs && (m.win_bs = rad_kv_block_size(b, m.win)) < 1) return RAD_E_INVAL;
        DS_TRY(rad_bind_layer_kv(b, l, m.win));
        DS_TRY(rad_kv_group_zero(b, m.win));
        const int64_t r = L.r;
        if (L.src) {
            const int64_t rows = r > 1 ? W : T;
            if (r > 1) {
                if (!m.tail.count(r) && !(m.tail[r] = tail_group(b, m, r))) return RAD_E_INVAL;
                /* Per-sequence state is kept on every rank (only paged groups zero): this rank's
                 * copy of the ring is small and never written. */
                DS_TRY(rad_bind_layer_kv(b, l, m.tail[r]));
            }
            const int ikf = kv_fmt(m.idx_kvf, c.idim, r), lkf = kv_fmt(m.lat_kvf, hd, r);
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", c.idim, r, ikf))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page, kv_fmt_name(ikf));
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            DS_TRY(rad_kv_group_zero(b, m.index[r]));
            if (!m.comp.count(r) && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", hd, r, lkf))) return RAD_E_INVAL;
            L.l_st = entry_store(b, rows, hd, r, m.lat, m.page, kv_fmt_name(lkf));
            if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
            DS_TRY(rad_kv_group_zero(b, m.comp[r]));
        }
        L.attn_fill = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_STR("dtype", "bf16"),
                                      RAD_F64("value", 0.0) },
                         {}, {}, { m.attn_out });
    } else {
        if (!L.cn_a && !L.pcn_a) L.norm_a = rmsnorm(b, m, T, L.attn_norm, D, m.xcol, m.xn);

        /* ---- attention: the query, the window key */
        L.wq_a = lw(b, m, nm.f("blk.%d.attn_q_a.weight", l), c.q_lora, D, grp);
        L.q_norm = w("blk.%d.attn_q_a_norm.weight", RAD_F32, { c.q_lora });
        L.wq_b = lw(b, m, nm.f("blk.%d.attn_q_b.weight", l), HD, c.q_lora, grp, RAD_SHARD_ROW,
                    m.ar * HD, (m.ar + 1) * HD);
        L.q_a = linear(b, m, T, L.wq_a, c.q_lora, D, m.xn, m.qr);
        L.q_an = rmsnorm(b, m, T, L.q_norm, c.q_lora, m.qr, m.qrn);
        L.q_b = linear(b, m, T, L.wq_b, HD, c.q_lora, m.qrn, m.q);
        const rad_buf cs = L.r > 0 ? m.cs_b : m.cs_a;
        L.q_rope = rope_rows(b, m, T, g.n_head, hd, 0, m.q, cs);
        L.wkv = lw(b, m, nm.f("blk.%d.attn_kv.weight", l), hd, D, grp);
        L.kv_norm = w("blk.%d.attn_kv_a_norm.weight", RAD_F32, { hd });
        L.kv_g = linear(b, m, T, L.wkv, hd, D, m.xn, m.kv);
        L.kv_f = norm_rope_rt(b, m, T, L.kv_norm, hd, 32, "fp8_e4m3", "ue8m0", m.kv, m.kvn, cs);
        if (!L.kv_f) {
            L.kv_n = rmsnorm(b, m, T, L.kv_norm, hd, m.kv, m.kvn);
            L.kv_rope = rope_rows(b, m, T, 1, hd, 0, m.kvn, cs);
            L.kv_q = roundtrip(b, T, hd, 32, "fp8_e4m3", "ue8m0", m.kvn);
        }
        if (!m.win && !(m.win = win_group(b, m))) return RAD_E_INVAL;
        L.kv_st = op(b, "kv_store",
                     { RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                       RAD_INT("block_size", 16), RAD_STR("kv_dtype", "bf16") },
                     {}, { m.kvn }, {});
        if (!m.win_bs) {
            m.win_bs = rad_kv_block_size(b, m.win);
            if (m.win_bs < 1) return RAD_E_INVAL;
        }
        DS_TRY(rad_bind_layer_kv(b, l, m.win));

        /* ---- the compressor and the index keys, at a kv source */
        const int64_t r = L.r;
        if (L.src) {
            const int64_t rows = r > 1 ? W : T;
            L.comp_norm = w("blk.%d.attn_compressor_norm.weight", RAD_F32, { hd });
            if (r > 1) {
                const char* cn = nm.f("blk.%d.attn_compressor_kvg.weight", l);
                DS_TRY(map_concat(b, cn, { nm.f("blk.%d.attn_compressor_kv.weight", l),
                                           nm.f("blk.%d.attn_compressor_gate.weight", l) }));
                L.comp_w = decl_w(b, cn, RAD_F32, { 2 * hd, D }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                                  grp);
                L.c_gemm = op(b, "gemm_nt_f32",
                              { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * hd), RAD_INT("K", D),
                                RAD_STR("dtype", "bf16") },
                              { L.comp_w }, { m.xn }, { m.kvs });
                if (!m.tail.count(r) && !(m.tail[r] = tail_group(b, m, r))) return RAD_E_INVAL;
                DS_TRY(rad_bind_layer_kv(b, l, m.tail[r]));
                L.c_work = op(b, "qsa_work",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", 2 * hd), RAD_INT("ratio", r),
                                RAD_INT("ring", r + g.max_spec), RAD_INT("seqs", g.max_seqs),
                                RAD_INT("work", W), RAD_STR("dtype", "f32") },
                              {}, { m.kvs }, { m.stage, m.page, m.bpos, m.nc });
                L.c_pool = op(b, "softmax_pool",
                              { RAD_RANGE("M", 1, W), RAD_INT("n", hd), RAD_INT("ratio", r),
                                RAD_STR("dtype", "bf16") },
                              {}, { m.stage, m.page }, { m.pooled });
                L.c_norm = rmsnorm(b, m, W, L.comp_norm, hd, m.pooled, m.lat);
            } else {
                L.comp_w = lw(b, m, nm.f("blk.%d.attn_compressor_kv.weight", l), hd, D, grp);
                L.c_gemm = linear(b, m, T, L.comp_w, hd, D, m.xn, m.pooled);
                L.c_norm = rmsnorm(b, m, T, L.comp_norm, hd, m.pooled, m.lat);
            }
            /* The compressed rows' angles, at each entry's first-token position. */
            L.c_tab = op(b, "rope_table",
                         { RAD_RANGE("M", 1, rows), RAD_INT("rot", c.rope), RAD_F64("theta", c.comp_theta),
                           RAD_F64("scale", 1.0), RAD_STR("dtype", "f32"),
                           RAD_F64("yarn_factor", c.yarn_factor > 1.0 ? c.yarn_factor : 0.0),
                           RAD_F64("yarn_low", c.yarn_low), RAD_F64("yarn_high", c.yarn_high),
                           RAD_F64("yarn_magnitude", 1.0) },
                         {}, { m.bpos }, { m.cs_b });
            /* Index keys from the unrotated latent, then the latent itself rotated and quantised. */
            L.ik_w = lw(b, m, nm.f("blk.%d.indexer.attn_k.weight", l), c.idim, hd, grp);
            L.ik_norm = w("blk.%d.indexer.k_norm.weight", RAD_F32, { c.idim });
            L.k_gemm = linear(b, m, rows, L.ik_w, c.idim, hd, m.lat, m.ik);
            L.k_f = norm_rope_rt(b, m, rows, L.ik_norm, c.idim, 32, "fp4_e2m1", "ue8m0", m.ik, m.ikn,
                                 m.cs_b);
            if (!L.k_f) {
                L.k_norm = rmsnorm(b, m, rows, L.ik_norm, c.idim, m.ik, m.ikn);
                L.k_rope = rope_rows(b, m, rows, 1, c.idim, 0, m.ikn, m.cs_b);
                L.k_q = roundtrip(b, rows, c.idim, 32, "fp4_e2m1", "ue8m0", m.ikn);
            }
            const int ikf = kv_fmt(m.idx_kvf, c.idim, r), lkf = kv_fmt(m.lat_kvf, hd, r);
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", c.idim, r, ikf))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page, kv_fmt_name(ikf));
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            L.l_f = rope_rt(b, m, rows, 1, hd, 0, 16, "fp4_e2m1", "e4m3", m.lat, m.cs_b);
            if (!L.l_f) {
                L.l_rope = rope_rows(b, m, rows, 1, hd, 0, m.lat, m.cs_b);
                L.l_q = roundtrip(b, rows, hd, 16, "fp4_e2m1", "e4m3", m.lat);
            }
            if (!m.comp.count(r) && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", hd, r, lkf))) return RAD_E_INVAL;
            L.l_st = entry_store(b, rows, hd, r, m.lat, m.page, kv_fmt_name(lkf));
            if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
            if (r > 1)
                L.c_tail = op(b, "qsa_tail_store",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", 2 * hd), RAD_INT("ratio", r),
                                RAD_INT("ring", r + g.max_spec), RAD_INT("seqs", g.max_seqs),
                                RAD_STR("dtype", "f32") },
                              {}, { m.kvs }, {});
        }

        /* ---- the indexer, at an index source */
        if (L.idx && r > 0) {
            L.iq_w = lw(b, m, nm.f("blk.%d.indexer.attn_q_b.weight", l), c.ih * c.idim, c.q_lora, grp);
            L.iproj = lw(b, m, nm.f("blk.%d.indexer.proj.weight", l), c.ih, D, grp);
            L.i_q = linear(b, m, T, L.iq_w, c.ih * c.idim, c.q_lora, m.qrn, m.iq);
            L.i_f = rope_rt(b, m, T, c.ih, c.idim, 0, 32, "fp4_e2m1", "ue8m0", m.iq, m.cs_b);
            if (!L.i_f) {
                L.i_rope = rope_rows(b, m, T, c.ih, c.idim, 0, m.iq, m.cs_b);
                L.i_qq = roundtrip(b, T, c.ih * c.idim, 32, "fp4_e2m1", "ue8m0", m.iq);
            }
            L.i_w = linear(b, m, T, L.iproj, c.ih, D, m.xn, m.iw);
            const int64_t width = g.max_ctx / r + 1, IR = m.irows;
            /* One index_score a run of rows (index_runs()): run p scores step rows [p R, p R + R)
             * into the score plane's first R rows. */
            const char* ikf = kv_fmt_name(kv_fmt(m.idx_kvf, c.idim, r));
            for (int64_t p = 0; p < m.iparts; ++p) {
                std::vector<RadParam> ps = { RAD_RANGE("M", 1, IR), RAD_INT("heads", c.ih), RAD_INT("dim", c.idim),
                                             RAD_INT("ratio", r), RAD_INT("width", width),
                                             RAD_F64("wscale", 1.0 / std::sqrt((double)c.idim * (double)c.ih)),
                                             RAD_STR("dtype", "bf16"), RAD_INT("row_part", p) };
                if (ikf) ps.push_back(RAD_STR("key_format", ikf));
                const rad_op h = op(b, "index_score", ps, {}, { m.iq, m.iw }, { m.score });
                if (!h) return RAD_E_INVAL;
                L.i_runs.push_back(h);
            }
            L.i_score = L.i_runs[0];
            if (l == c.cand) {
                if (!m.keep) return RAD_E_INVAL;
                L.i_cand = op(b, "index_candidates",
                              { RAD_RANGE("M", 1, IR), RAD_INT("width", width),
                                RAD_INT("block", c.cand_block), RAD_INT("topk_blocks", c.cand_blocks),
                                RAD_INT("ratio", r) },
                              {}, { m.score }, { m.keep });
                if (!L.i_cand) return RAD_E_INVAL;
            }
            const bool uses = c.cand >= 0 && c.cand < l;
            if (uses && (!m.keep || c.ratio[(size_t)c.cand] != r)) {
                fprintf(stderr, "radiance: deepseek41: layer %d reads the candidate layer's blocks at "
                                "another ratio (%lld) or there are none\n", l, (long long)r);
                return RAD_E_UNSUPPORTED;
            }
            std::vector<RadParam> tp = { RAD_RANGE("M", 1, IR), RAD_INT("width", width),
                                         RAD_INT("topk", c.itopk), RAD_INT("ratio", r) };
            if (uses) tp.push_back(RAD_INT("block", c.cand_block));
            L.i_topk = op(b, "index_topk", tp, {},
                          uses ? std::initializer_list<rad_buf>{ m.score, m.keep }
                               : std::initializer_list<rad_buf>{ m.score },
                          { m.sel });
            if (!L.i_topk) return RAD_E_INVAL;
        }

        /* ---- the sparse attention and the output projection */
        L.sinks = wts(b, g, nm.f("blk.%d.attn_sinks.weight", l), RAD_F32, { g.n_head }, grp, RAD_SHARD_ROW,
                      m.ar * g.n_head, (m.ar + 1) * g.n_head);
        std::vector<RadParam> aps = { RAD_RANGE("M", 1, T), RAD_INT("heads", g.n_head), RAD_INT("dim", hd),
                                      RAD_INT("window", c.window), RAD_INT("ratio", r),
                                      RAD_INT("topk", r > 0 ? c.itopk : 0),
                                      RAD_F64("scale", 1.0 / std::sqrt((double)hd)), RAD_STR("dtype", "bf16") };
        if (const char* lkf = r > 0 ? kv_fmt_name(kv_fmt(m.lat_kvf, hd, r)) : nullptr)
            aps.push_back(RAD_STR("latent_format", lkf));
        L.attn = op(b, "attn_sparse_sink", aps,
                    { L.sinks },
                    r > 0 ? std::initializer_list<rad_buf>{ m.q, m.sel }
                          : std::initializer_list<rad_buf>{ m.q },
                    { m.o });
        L.o_f = rope_rt(b, m, T, g.n_head, hd, 1, 32, "fp8_e4m3", "ue8m0", m.o, cs);
        if (!L.o_f) {
            L.o_rope = rope_rows(b, m, T, g.n_head, hd, 1, m.o, cs);
            L.o_q = roundtrip(b, T, HD, 32, "fp8_e4m3", "ue8m0", m.o);
        }
        /* This rank's output groups, each a block of the block-diagonal wo_a. */
        const int64_t gd = HD / m.gr;
        for (int j = 0; j < m.gr; ++j) {
            const int gi_ = (int)(m.ar * m.gr + j);
            const char* n = nm.f("blk.%d.attn_output_a.g%d.weight", l, gi_);
            DS_TRY(map_rows(b, n, { { nm.f("blk.%d.attn_output_a.weight", l), (int)(gi_ * c.o_rank),
                                            c.o_rank } }));
            L.wo_a.push_back(lw(b, m, n, c.o_rank, gd, grp, RAD_SHARD_NONE, 0, 0, true));
            if (!L.wo_a.back()) return RAD_E_INVAL;
            L.o_a.push_back(linear(b, m, T, L.wo_a.back(), c.o_rank, gd, m.o, m.oa));
        }
        const int64_t ok = m.gr * c.o_rank;
        L.wo_b = lw(b, m, nm.f("blk.%d.attn_output_b.weight", l), D, ok, grp, RAD_SHARD_COL,
                    m.ar * ok, (m.ar + 1) * ok);
        L.o_b = linear(b, m, T, L.wo_b, D, ok, m.oa, m.attn_out);
    }
    /* The heads' partial outputs (zeros on an attention-zero rank) summed over every rank. */
    if (g.world > 1)
        L.ar_attn = op(b, "all_reduce",
                       { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                         RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                         RAD_INT("min_bytes", g.wire_min_bytes) },
                       {}, { m.attn_out }, { m.attn_out });
    L.post_a = op(b, "mhc_post", hcp, {}, { m.attn_out, m.h0, m.post, m.comb }, { m.h1 });

    /* ---- the ffn connection and the MoE */
    L.hc_f_fn = w("blk.%d.hc_ffn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_f_scale = w("blk.%d.hc_ffn_scale.weight", RAD_F32, { 3 });
    L.hc_f_base = w("blk.%d.hc_ffn_base.weight", RAD_F32, { mix });
    L.mix_f = op(b, "mhc_mix", mixp, { L.hc_f_fn, L.hc_f_scale, L.hc_f_base }, { m.h1 },
                 { m.pre_f, m.post, m.comb });
    L.ffn_norm = w("blk.%d.ffn_norm.weight", RAD_F32, { D });
    L.pcn_f = post_collapse_norm(b, m, T, L.ffn_norm, m.attn_out, m.h0, m.h1, m.pre_a, m.xcol,
                                 m.act.x);
    if (!L.pcn_f) L.cn_f = collapse_norm(b, m, T, L.ffn_norm, m.h1, m.pre_a, m.xcol, m.act.x);
    if (!L.pcn_f && !L.cn_f) {
        L.col_f = op(b, "mhc_collapse", hcp, {}, { m.h1, m.pre_a }, { m.xcol });
        L.norm_f = rmsnorm(b, m, T, L.ffn_norm, D, m.xcol, m.act.x);
    }
    MoeFP8::Src s;
    s.norm = nullptr;
    moe_names(m, l, &s);
    s.cfg.n_expert_all = c.n_expert;
    s.cfg.n_expert = m.ne_rank;
    s.cfg.expert_base = m.ne_base;
    s.cfg.probed_expert = s.cfg.expert_base;
    s.cfg.top_k = c.top_k;
    s.cfg.n_ff_exp = c.n_ff;
    s.cfg.n_ff_shared = g.n_ff;
    s.cfg.shared_lo = m.sh_lo;
    s.cfg.shared_hi = m.sh_hi;
    s.cfg.rows = T;
    s.cfg.norm_topk = c.norm_topk;
    s.cfg.sigmoid_router = 1;
    s.cfg.router_score = "sqrtsoftplus";
    s.cfg.route_scale = c.route_scale;
    s.cfg.swiglu_limit = c.swiglu_limit;
    s.cfg.shared_ungated = 1;
    s.cfg.calib = (m.calib_hi > m.calib_lo && l >= m.calib_lo && l < m.calib_hi) ? 1 : 0;
    DS_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
    DS_TRY(L.moe.declare(b, nm, g, l, m.mw, s, false, false));
    L.post_f = op(b, "mhc_post", hcp, {}, { m.act.x, m.h1, m.post, m.comb }, { m.h0 });
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 ||
        ctx->slot_base < 0 || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->max_spec > 0 || ctx->tp_wire_lossy ||
        (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16)) {
        fprintf(stderr, "radiance: deepseek41 requires BF16 KV, exact collectives and draft off\n");
        return RAD_E_UNSUPPORTED;
    }
    auto& slot = ctx->shape_probe ? probe_model : models[rad_build_slot(ctx)];
    slot = std::make_unique<Model>();
    auto& m = *slot;
    m.nm = Names(ctx->scope);
    auto& nm = m.nm;
    DS_TRY(read_cfg(m.c, meta));
    DS_TRY(geometry(m, ctx));
    const auto& c = m.c;
    auto& g = m.g;
    int64_t rmin = 0;
    for (int l = 0; l < c.n_layer; ++l)
        if (c.ratio[(size_t)l] > 0 && (rmin == 0 || c.ratio[(size_t)l] < rmin)) rmin = c.ratio[(size_t)l];
    m.work = g.max_tok + g.max_seqs;
    m.cmax = g.max_ctx / std::max<int64_t>(rmin, 1) + 1;
    index_runs(m);
    DS_TRY(buffers(b, m));
    const int64_t T = g.max_tok, D = c.n_embd;
    /* The vocabulary is split over the ranks (shard_extent): the lookup gives zeros outside this
     * rank's rows and an all-reduce completes it; the head's logits stay sharded for the sampler. */
    const int64_t v0 = g.vocab_off, v1 = g.vocab_off + g.n_vocab;
    m.token = lw(b, m, nm.f("token_embd.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1,
                 false, RAD_ACCESS_VOCAB);
    m.head = lw(b, m, nm.f("output.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1, false,
                RAD_ACCESS_VOCAB);
    if (!m.token || !m.head) return RAD_E_INVAL;
    m.final_gain = wt(b, nm.f("output_norm.weight"), RAD_F32, { D }, grp_model());
    /* A GGUF's table in llama.cpp's blocks is gathered from them (embed_lookup_ggml). */
    const auto tt = m.wty.find(m.token);
    if (tt != m.wty.end())
        m.embed = op(b, "embed_lookup_ggml",
                     { RAD_RANGE("M", 1, T), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("type", tt->second), RAD_STR("dtype", "bf16"),
                       RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { m.xcol });
    else
        m.embed = op(b, "embed_lookup",
                     { RAD_RANGE("M", 1, T), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { m.xcol });
    if (g.world > 1)
        m.embed_ar = op(b, "all_reduce",
                        { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                          RAD_STR("dtype", "bf16"), RAD_INT("exact", 1) },
                        {}, { m.xcol }, { m.xcol });
    m.expand = op(b, "cast",
                  { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_STR("from", "bf16"),
                    RAD_STR("to", "bf16") },
                  {}, { m.xcol }, { m.h0 });
    /* Layer 0's attention collapses with one-hot gates: stream 0. */
    m.pre0 = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", c.hc), RAD_STR("dtype", "f32"),
                             RAD_F64("value", 0.0) },
                {}, {}, { m.pre_f });
    m.pre1 = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                             RAD_F64("value", 1.0) },
                {}, {}, { m.pre_f });
    m.rope_a = op(b, "rope_table",
                  { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.theta),
                    RAD_F64("scale", 1.0), RAD_STR("dtype", "f32") },
                  {}, {}, { m.cs_a });
    m.rope_b_step = op(b, "rope_table",
                       { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.comp_theta),
                         RAD_F64("scale", 1.0), RAD_STR("dtype", "f32"),
                         RAD_F64("yarn_factor", c.yarn_factor > 1.0 ? c.yarn_factor : 0.0),
                         RAD_F64("yarn_low", c.yarn_low), RAD_F64("yarn_high", c.yarn_high),
                         RAD_F64("yarn_magnitude", 1.0) },
                       {}, {}, { m.cs_b });
    if (!c.ple.empty()) {
        m.tmap = wt(b, "deepseek41.engram.token_map", RAD_I32, { c.n_vocab }, grp_model());
        /* Bf16, or a single-level affine encoding (codes, a scale per group, an optional
         * codebook: a GGUF's IQ4_NL, Q8_0, Q4_0) read in place from its stored planes, since the
         * table (103 GiB in the 1017) is too large to relayout or copy; it stays on whatever tier
         * holds the container's bytes. */
        const char* tn = nm.f("per_layer_token_embd.weight");
        RadEncoding te{};
        const bool known = weight_enc(b, tn, &te);
        if (!known || (rad_enc_is(&te, "plain") && te.n_planes == 1)) {
            m.ple_table = decl_w(b, tn, RAD_BF16, { c.ple_rows, c.ple_dim }, RAD_ACCESS_VOCAB,
                                 RAD_SHARD_NONE, grp_model());
            if (!m.ple_table) return RAD_E_INVAL;
        } else {
            const RadEncPlane* cp = rad_enc_plane(&te, "codes");
            const RadEncPlane* sp = rad_enc_plane(&te, "scale");
            const RadEncPlane* tp = rad_enc_plane(&te, "table");
            const int64_t G = sp ? sp->block[1] : 0;
            if (!rad_enc_is(&te, "affine") || te.transform[0] || !cp || !sp || cp->block[0] != 1 ||
                cp->block[1] != 1 || sp->block[0] != 1 || G <= 0 || c.ple_dim % G ||
                te.n_planes != 2 + (tp ? 1 : 0)) {
                char en[256];
                rad_enc_format(&te, en, sizeof en);
                fprintf(stderr, "radiance: deepseek41: the Engram table is %s; it is read as bf16 or "
                                "as codes with one scale per group of a row and an optional "
                                "codebook\n", en);
                return RAD_E_DTYPE;
            }
            m.ple_group = G;
            m.ple_codes = decl_view(b, tn, nullptr, "codes", cp->dtype, { c.ple_rows, c.ple_dim },
                                    RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
            m.ple_scale = decl_view(b, nm.f("per_layer_token_embd.scale"), tn, "scale", sp->dtype,
                                    { c.ple_rows, c.ple_dim / G }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                                    grp_model());
            if (tp)
                m.ple_tab = decl_view(b, nm.f("per_layer_token_embd.table"), tn, "table", tp->dtype,
                                      { tp->extent[0], tp->extent[1] }, RAD_ACCESS_PER_TOKEN,
                                      RAD_SHARD_NONE, grp_model());
            if (!m.ple_codes || !m.ple_scale || (tp && !m.ple_tab)) return RAD_E_INVAL;
        }
        m.tok_map = op(b, "token_map", { RAD_RANGE("M", 1, T), RAD_INT("n_vocab", c.n_vocab) },
                       { m.tmap }, {}, { m.ctok });
    }
    m.layers.resize((size_t)c.n_layer);
    if (ctx->calib_dir && ctx->calib_dir[0]) {
        m.calib_lo = ctx->calib_layer_lo;
        m.calib_hi = ctx->calib_layer_hi > ctx->calib_layer_lo ? ctx->calib_layer_hi : (int)c.n_layer;
    }
    for (int l = 0; l < c.n_layer; ++l) DS_TRY(layer(b, m, m.layers[(size_t)l], l));
    m.pcn_fin = post_collapse_norm(b, m, T, m.final_gain, m.act.x, m.h1, m.h0, m.pre_f, m.xcol,
                                   m.final_hidden);
    if (!m.pcn_fin)
        m.cn_fin = collapse_norm(b, m, T, m.final_gain, m.h0, m.pre_f, m.xcol, m.final_hidden);
    if (!m.pcn_fin && !m.cn_fin) {
        m.contract = op(b, "mhc_collapse",
                        { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                        {}, { m.h0, m.pre_f }, { m.xcol });
        m.fnorm = rmsnorm(b, m, T, m.final_gain, D, m.xcol, m.final_hidden);
    }
    m.gather = op(b, "gather_rows",
                  { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", D), RAD_STR("dtype", "bf16") },
                  {}, { m.final_hidden }, { m.logit_input });
    const auto ht = m.wty.find(m.head);
    if (ht != m.wty.end())
        m.head_op = op(b, "logits_gemm_ggml",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("type", ht->second), RAD_STR("dtype", "bf16") },
                       { m.head }, { m.logit_input }, { m.logits });
    else
        m.head_op = op(b, "logits_gemm",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                       { m.head }, { m.logit_input }, { m.logits });
    rad_note(b, "DeepSeek-V4.1: %lld layers, hc %lld, window %lld, %zu kv sources, %zu index "
                "sources, %lld experts top-%lld, Engram at %zu layers",
             (long long)c.n_layer, (long long)c.hc, (long long)c.window, c.kv_src.size(),
             c.idx_src.size(), (long long)c.n_expert, (long long)c.top_k, c.ple.size());
    if (g.world > 1 && m.ne_rank * g.world != c.n_expert)
        rad_note(b, "routed experts split unevenly (RADIANCE_EXPERT_SPLIT): rank %d holds %lld of %lld a layer, from %lld",
                 g.rank, (long long)m.ne_rank, (long long)c.n_expert, (long long)m.ne_base);
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
struct MixBeside {
    const Layer* L;
    const Model* m;
    int64_t T;
};
static void mix_f_beside(RadCtx* x, void* a) {
    const MixBeside& p = *(const MixBeside*)a;
    const Layer& L = *p.L;
    const Model& m = *p.m;
    const int64_t T = p.T;
    RAD_ISSUE_N(x, L.mix_f, T, brows(m.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale),
                RAD_W(L.hc_f_base), brows(m.pre_f, T), brows(m.post, T), brows(m.comb, T));
}

static void step(RadCtx* x, const RadBatch* batch) {
    auto& m = *models[rad_slot(x)];
    if (batch->draft_pass != 0) {
        rad_step_fail(x, "deepseek41: draft passes are unsupported");
        return;
    }
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const auto* wb = kv_batch(batch, m.win);
    if (!wb || !wb->block_table || !wb->slot_mapping) {
        rad_step_fail(x, "deepseek41: the window group has no block table");
        return;
    }
    const RadOperand wbt = praw2(wb->block_table, RAD_I32, S, wb->block_table_pitch);
    /* The window group's table is rebased at the sequence's first held block; its seqused counts
     * from there, and attn_sparse_sink recovers the base from it (docs/OPS.md). */
    const RadOperand wused = praw(wb->seqused, RAD_I32, S);

    RAD_ISSUE_N(x, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.xcol, T));
    if (m.embed_ar) RAD_ISSUE_N(x, m.embed_ar, T * D, brows(m.xcol, T), RAD_NONE);
    for (int64_t i = 0; i < c.hc; ++i)
        RAD_ISSUE_N(x, m.expand, T, brows(m.xcol, T), bcol(m.h0, i * D, D, T));
    RAD_ISSUE_N(x, m.pre0, T, brows(m.pre_f, T));
    RAD_ISSUE_N(x, m.pre1, T, bcol(m.pre_f, 0, 1, T));
    RAD_ISSUE_N(x, m.rope_a, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, m.rope_b_step, T, pos, brows(m.cs_b, g.max_ctx));
    if (m.tok_map)
        RAD_ISSUE_N(x, m.tok_map, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tmap),
                    brows(m.ctok, T));

    /* The Engram's row ids are a function of the batch's tokens alone (token_map, each Engram
     * layer's token history), never of an activation, so every Engram layer's ngram_ids is issued
     * here, before the first block, each into its own host buffer (rad_block_ple.h's ids() does
     * the same). A host-site gather (the table on the disk or RAM tier) joins on the last device
     * op that touched the host arena: these few ops, or the previous Engram layer's copy of its
     * rows. It then reads its rows while the device runs the blocks the host has already issued,
     * where ngram_ids at its layer (and device-domain rows) made the device drain before every
     * gather and idle through it. */
    for (auto& L : m.layers) {
        if (!L.e_ids) continue;
        const int l = L.l;
        const auto* tk = kv_batch(batch, m.tok);
        const int32_t* tk_idx = tk ? tk->state_index : nullptr;
        int64_t Dn = 0, DT = 0;
        batch_split(batch, &Dn, &DT);
        const int64_t P = S - Dn;
        if (Dn > 0)
            RAD_ISSUE_N(x, L.e_ids, DT, brows(m.ctok, DT), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens, RAD_I32, Dn + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off), praw(tk_idx, RAD_I32, Dn),
                        praw(batch->ctx_lens, RAD_I32, Dn),
                        praw(batch->num_accepted, RAD_I32, Dn), brows(L.ids, DT), RAD_NONE);
        if (P > 0)
            RAD_ISSUE_N(x, L.e_ids, T, brows(m.ctok, T), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens + Dn, RAD_I32, P + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off),
                        praw(tk_idx ? tk_idx + Dn : nullptr, RAD_I32, P),
                        praw(batch->ctx_lens + Dn, RAD_I32, P), RAD_NONE, brows(L.ids, T),
                        RAD_NONE);
    }

    for (auto& L : m.layers) {
        const int l = L.l;
        const int64_t r = L.r;
        if (L.e_ids) {
            const int64_t H = c.ple_heads;
            /* T rows of operands; the gather's band is T * H, one table row per head per token,
             * read through the ids and rows as flat (rad_block_ple.h does the same). */
            if (m.ple_group > 0)
                RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_codes), RAD_W(m.ple_scale),
                            m.ple_tab ? RAD_W(m.ple_tab) : RAD_NONE, brows(m.emb, T));
            else
                RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_table), brows(m.emb, T));
            RAD_ISSUE_N(x, L.e_copy, T, brows(m.emb, T), brows(m.embd, T));
            lin_issue(x, m, L.e_gemm, T, brows(m.embd, T), L.e_kv, brows(m.ekv, T));
            RAD_ISSUE_N(x, L.e_gate_op, T, brows(m.h0, T), bcol(m.ekv, 0, c.hc * D, T),
                        bcol(m.ekv, c.hc * D, D, T), RAD_W(L.e_gate), brows(m.h1, T));
            RAD_ISSUE_N(x, L.e_back, T, brows(m.h1, T), brows(m.h0, T));
        }
        /* Issue order of the connections' mixes (Ctx::haz_mark: a recorded pass leaves a dispatch
         * unordered when it touches nothing in flight). Each connection's collapse reads the gates
         * the previous connection's mix wrote, never what its own mix writes, and nothing reads a
         * mix's output before the next mhc_post. So, same kernels on the same operands:
         * - the collapse goes first. Where it is its own op (no fused mhc_post before it: the
         *   first layer, after an Engram block, the attention-zero rank) the mix's second launch,
         *   ordered anyway, is then the barrier the norm needs;
         * - on an attention rank the attention mix goes between the output projection's groups
         *   and attn_output_b, where its launches overlap those GEMVs instead of running nearly
         *   alone at the layer's start; on the attention-zero rank it follows the collapse. */
        const bool late = !m.zero;
        const auto mix_a = [&] {
            RAD_ISSUE_N(x, L.mix_a, T, brows(m.h0, T), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                        RAD_W(L.hc_a_base), brows(m.pre_a, T), brows(m.post, T), brows(m.comb, T));
        };
        const auto col_a = [&] {
            if (L.cn_a)
                RAD_ISSUE_N(x, L.cn_a, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                            RAD_W(L.attn_norm), brows(m.xn, T));
            else if (L.col_a)   /* 0 under pcn_a: issued for the previous post_f */
                RAD_ISSUE_N(x, L.col_a, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T));
        };
        col_a();
        if (!late) mix_a();
        if (m.zero) {
            RAD_ISSUE_N(x, L.attn_fill, T, brows(m.attn_out, T));
        } else {
            if (L.norm_a)
                RAD_ISSUE_N(x, L.norm_a, T, brows(m.xcol, T), RAD_W(L.attn_norm), brows(m.xn, T));
            const rad_buf cs = r > 0 ? m.cs_b : m.cs_a;
            /* The compressor and indexer reuse transient arena storage. Their issue order must
             * follow declaration order so a later projection cannot overwrite the staging
             * plane while the pool still needs it. */
            const bool ctail = L.src && r > 1;
            int64_t rows = T, tpitch = 1;
            RadOperand page = RAD_NONE, bpos = pos;
            const auto* cb = L.src ? kv_batch(batch, m.comp[r]) : nullptr;
            const auto* tb = ctail ? kv_batch(batch, m.tail[r]) : nullptr;
            if (L.src) {
                const auto* ib = kv_batch(batch, m.index[r]);
                if (!cb || !ib || !cb->block_table || !ib->block_table) {
                    rad_step_fail(x, "deepseek41: a compressed group has no block table");
                    return;
                }
                page = praw(cb->slot_mapping, RAD_I32, T);
                if (ctail) {
                    if (!tb || !tb->state_index) {
                        rad_step_fail(x, "deepseek41: the compressor tail has no state slots");
                        return;
                    }
                    tpitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
                    rows = m.work;
                    page = brows(m.page, rows);
                    bpos = brows(m.bpos, rows);
                }
            }
            const auto q_a = [&] { lin_issue(x, m, L.q_a, T, brows(m.xn, T), L.wq_a, brows(m.qr, T)); };
            const auto q_an = [&] { RAD_ISSUE_N(x, L.q_an, T, brows(m.qr, T), RAD_W(L.q_norm), brows(m.qrn, T)); };
            const auto q_b = [&] { lin_issue(x, m, L.q_b, T, brows(m.qrn, T), L.wq_b, brows(m.q, T)); };
            const auto q_rope = [&] { RAD_ISSUE_N(x, L.q_rope, T, brows(m.q, T), brows(cs, g.max_ctx), pos); };
            const auto kv_g = [&] { lin_issue(x, m, L.kv_g, T, brows(m.xn, T), L.wkv, brows(m.kv, T)); };
            /* The window key's norm, rope and round trip: one op (kv_f) where declared, else
             * kv_n here and kv_rope / kv_q at the next two points. */
            const auto kv_n = [&] {
                if (L.kv_f)
                    RAD_ISSUE_N(x, L.kv_f, T, brows(m.kv, T), RAD_W(L.kv_norm), brows(m.kvn, T),
                                brows(cs, g.max_ctx), pos);
                else
                    RAD_ISSUE_N(x, L.kv_n, T, brows(m.kv, T), RAD_W(L.kv_norm), brows(m.kvn, T));
            };
            const auto kv_rope = [&] {
                if (!L.kv_f) RAD_ISSUE_N(x, L.kv_rope, T, brows(m.kvn, T), brows(cs, g.max_ctx), pos);
            };
            const auto kv_q = [&] {
                if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, T, brows(m.kvn, T));
            };
            const auto kv_st = [&] {
                RAD_ISSUE_N(x, L.kv_st, T, bcol(m.kvn, 0, hd / 2, T), bcol(m.kvn, hd / 2, hd / 2, T),
                            praw(wb->slot_mapping, RAD_I32, T), kv_cache(m.win, l));
            };
            const auto c_gemm = [&] {
                if (L.src)
                    RAD_ISSUE_N(x, L.c_gemm, T, brows(m.xn, T), RAD_W(L.comp_w),
                                brows(ctail ? m.kvs : m.pooled, T));
            };
            const auto c_work = [&] {
                if (ctail)
                    RAD_ISSUE_N(x, L.c_work, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, tpitch),
                                praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), RAD_NONE,
                                brows(m.stage, rows * r), brows(m.page, rows), brows(m.bpos, rows),
                                brows(m.nc, S));
            };
            const auto c_pool = [&] {
                if (ctail)
                    RAD_ISSUE_N(x, L.c_pool, rows, brows(m.stage, rows * r), brows(m.pooled, rows),
                                brows(m.page, rows));
            };
            const auto c_norm = [&] {
                if (L.src)
                    RAD_ISSUE_N(x, L.c_norm, rows, brows(m.pooled, rows), RAD_W(L.comp_norm),
                                brows(m.lat, rows));
            };
            const auto c_tab = [&] {
                if (L.src) RAD_ISSUE_N(x, L.c_tab, rows, bpos, brows(m.cs_b, g.max_ctx));
            };
            const auto k_gemm = [&] {
                if (L.src) lin_issue(x, m, L.k_gemm, rows, brows(m.lat, rows), L.ik_w, brows(m.ik, rows));
            };
            /* The index key's norm, rope and round trip (k_f where declared); the latent's rope and
             * round trip (l_f where declared); the index query's (i_f where declared). Each part
             * function issues the fused op at the chain's first point and nothing at the others. */
            const auto k_norm = [&] {
                if (!L.src) return;
                if (L.k_f)
                    RAD_ISSUE_N(x, L.k_f, rows, brows(m.ik, rows), RAD_W(L.ik_norm), brows(m.ikn, rows),
                                brows(m.cs_b, g.max_ctx), bpos);
                else
                    RAD_ISSUE_N(x, L.k_norm, rows, brows(m.ik, rows), RAD_W(L.ik_norm), brows(m.ikn, rows));
            };
            const auto k_rope = [&] {
                if (L.src && !L.k_f) RAD_ISSUE_N(x, L.k_rope, rows, brows(m.ikn, rows), brows(m.cs_b, g.max_ctx), bpos);
            };
            const auto k_q = [&] {
                if (L.src && L.k_q) RAD_ISSUE_N(x, L.k_q, rows, brows(m.ikn, rows));
            };
            const auto k_st = [&] {
                if (L.src) RAD_ISSUE_N(x, L.k_st, rows, brows(m.ikn, rows), page, kv_cache(m.index[r], l));
            };
            const auto l_rope = [&] {
                if (!L.src) return;
                if (L.l_f)
                    RAD_ISSUE_N(x, L.l_f, rows, brows(m.lat, rows), brows(m.cs_b, g.max_ctx), bpos);
                else
                    RAD_ISSUE_N(x, L.l_rope, rows, brows(m.lat, rows), brows(m.cs_b, g.max_ctx), bpos);
            };
            const auto l_q = [&] {
                if (L.src && L.l_q) RAD_ISSUE_N(x, L.l_q, rows, brows(m.lat, rows));
            };
            const auto l_st = [&] {
                if (L.src) RAD_ISSUE_N(x, L.l_st, rows, brows(m.lat, rows), page, kv_cache(m.comp[r], l));
            };
            const auto c_tail = [&] {
                if (ctail && L.c_tail)
                    RAD_ISSUE_N(x, L.c_tail, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, tpitch));
            };
            const auto i_q = [&] {
                if (L.i_score) lin_issue(x, m, L.i_q, T, brows(m.qrn, T), L.iq_w, brows(m.iq, T));
            };
            const auto i_rope = [&] {
                if (!L.i_score) return;
                if (L.i_f)
                    RAD_ISSUE_N(x, L.i_f, T, brows(m.iq, T), brows(m.cs_b, g.max_ctx), pos);
                else
                    RAD_ISSUE_N(x, L.i_rope, T, brows(m.iq, T), brows(m.cs_b, g.max_ctx), pos);
            };
            const auto i_qq = [&] {
                if (L.i_score && L.i_qq) RAD_ISSUE_N(x, L.i_qq, T, brows(m.iq, T));
            };
            const auto i_w = [&] {
                if (L.i_score) lin_issue(x, m, L.i_w, T, brows(m.xn, T), L.iproj, brows(m.iw, T));
            };
            const auto i_score = [&] {   /* the scores, the candidates and the selection */
                if (!L.i_score) return;
                const int src = c.src_of[(size_t)l];
                const auto* ib = kv_batch(batch, m.index[r]);
                const int64_t live = std::min<int64_t>(
                    g.max_ctx / r + 1, ((int64_t)batch->max_ctx_len + batch->max_q_len) / r + 1);
                const int64_t kw = keep_words(live, c.cand_block);
                const bool uses = c.cand >= 0 && c.cand < l;
                /* Run by run (index_runs()): score R rows into the plane, then select from them
                 * before the next run overwrites it. Rows are independent, so each row's score,
                 * candidates and selection are the whole step's. */
                const int64_t n = (T + m.irows - 1) / m.irows, R = (T + n - 1) / n;
                for (int64_t p = 0; p * R < T; ++p) {
                    const int64_t r0 = p * R, rows = std::min(R, T - r0);
                    const RadOperand prow = praw(batch->positions + r0, RAD_I32, rows);
                    RAD_ISSUE_N(x, L.i_runs[(size_t)p], R, brows(m.iq, T), brows(m.iw, T),
                                kv_cache(m.index[r], src),
                                praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), pos, cu,
                                bcol(m.score, 0, live, R));
                    if (L.i_cand)
                        RAD_ISSUE_N(x, L.i_cand, rows, bcol(m.score, 0, live, rows), prow,
                                    bcol_at(m.keep, r0, m.kwords, 0, kw, rows));
                    RAD_ISSUE_N(x, L.i_topk, rows, bcol(m.score, 0, live, rows),
                                uses ? bcol_at(m.keep, r0, m.kwords, 0, kw, rows) : RAD_NONE, prow,
                                brow_slice(m.sel, r0, rows, c.itopk));
                }
            };

            q_a();
            q_an();
            q_b();
            q_rope();
            kv_g();
            kv_n();
            kv_rope();
            kv_q();
            kv_st();
            c_gemm();
            c_work();
            c_pool();
            c_norm();
            c_tab();
            k_gemm();
            k_norm();
            k_rope();
            k_q();
            k_st();
            l_rope();
            l_q();
            l_st();
            c_tail();
            i_q();
            i_rope();
            i_qq();
            i_w();
            i_score();

            if (r > 0) {
                const int src = c.src_of[(size_t)l];
                const auto* cb = kv_batch(batch, m.comp[r]);
                RAD_ISSUE_N(x, L.attn, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused,
                            kv_cache(m.comp[r], src),
                            praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), brows(m.sel, T),
                            pos, cu, RAD_W(L.sinks), brows(m.o, T));
            } else {
                RAD_ISSUE_N(x, L.attn, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused, RAD_NONE,
                            RAD_NONE, RAD_NONE, pos, cu, RAD_W(L.sinks), brows(m.o, T));
            }
            if (L.o_f) {
                RAD_ISSUE_N(x, L.o_f, T, brows(m.o, T), brows(cs, g.max_ctx), pos);
            } else {
                RAD_ISSUE_N(x, L.o_rope, T, brows(m.o, T), brows(cs, g.max_ctx), pos);
                if (L.o_q) RAD_ISSUE_N(x, L.o_q, T, brows(m.o, T));
            }
            const int64_t gd = HD / m.gr;
            for (int j = 0; j < m.gr; ++j)
                lin_issue(x, m, L.o_a[(size_t)j], T, bcol(m.o, j * gd, gd, T), L.wo_a[(size_t)j],
                          bcol(m.oa, j * c.o_rank, c.o_rank, T));
            if (late) mix_a();
            lin_issue(x, m, L.o_b, T, brows(m.oa, T), L.wo_b, brows(m.attn_out, T));
        }
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, T * D, brows(m.attn_out, T), RAD_NONE);
        if (L.pcn_f)   /* post_a, col_f and norm_f in one */
            RAD_ISSUE_N(x, L.pcn_f, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                        RAD_W(L.ffn_norm), brows(m.act.x, T));
        else
            RAD_ISSUE_N(x, L.post_a, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T));

        /* The ffn connection's mix is likewise needed first by the next mhc_post, so the MoE block
         * issues it (MoeFP8::Beside) behind its routed and shared gate_up, where a recorded pass
         * overlaps it with those GEMMs instead of with the collapse and the norm (or, under the
         * fused post, with nothing). It reads h1 and writes what post_a (fused or not) has read. */
        const auto col_f = [&] {
            if (L.pcn_f) {
            } else if (L.cn_f) {
                RAD_ISSUE_N(x, L.cn_f, T, brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                            RAD_W(L.ffn_norm), brows(m.act.x, T));
            } else {
                RAD_ISSUE_N(x, L.col_f, T, brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T));
                RAD_ISSUE_N(x, L.norm_f, T, brows(m.xcol, T), RAD_W(L.ffn_norm), brows(m.act.x, T));
            }
        };
        col_f();
        MixBeside mb{ &L, &m, T };
        MoeFP8::Beside bs;
        bs.fn = mix_f_beside;
        bs.arg = &mb;
        L.moe.step(x, batch, bs);
        /* post_f, or post_f with the next collapse + norm (the next layer's pcn_a, or pcn_fin). */
        const Layer* nx = L.l + 1 < c.n_layer ? &m.layers[(size_t)L.l + 1] : nullptr;
        if (nx && nx->pcn_a)
            RAD_ISSUE_N(x, nx->pcn_a, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(nx->attn_norm), brows(m.xn, T));
        else if (!nx && m.pcn_fin)
            RAD_ISSUE_N(x, m.pcn_fin, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(m.final_gain), brows(m.final_hidden, T));
        else
            RAD_ISSUE_N(x, L.post_f, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T));
    }
    if (m.pcn_fin) {
    } else if (m.cn_fin) {
        RAD_ISSUE_N(x, m.cn_fin, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                    RAD_W(m.final_gain), brows(m.final_hidden, T));
    } else {
        RAD_ISSUE_N(x, m.contract, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T));
        RAD_ISSUE_N(x, m.fnorm, T, brows(m.xcol, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    }
    if (batch->n_out > 0) {
        RAD_ISSUE_N(x, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(x, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    Cfg cfg;
    const int result = read_cfg(cfg, meta);
    if (result != RAD_OK) return result;
    out->shape_probe_ok = 1;
    return RAD_OK;
}

/* DSML tool calls with typed argument tags, which the template analysis cannot state. */
static const RadChatFormat* chat_format(const RadModelMeta*) { return rad::arch::deepseek41_chat_format(); }
}  // namespace deepseek41

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(deepseek41)
RAD_ARCH_CHAT_FORMAT(deepseek41)
RAD_ARCH_PLUGIN(deepseek41, "deepseek41", "", "0.1.0",
                "DeepSeek-V4.1: hyper-connection streams, window + sparse compressed latent "
                "attention, sqrt(softplus) MoE, Engram")
#endif
