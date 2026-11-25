/* inkling2_dspark.h -- Inkling 2's own DSpark drafter (`mtp.safetensors`, `mtp_config` in the
 * release's config.json), served by arch/inkling2 through the engine's block drafter.
 *
 * ============================== THE SHAPE ==============================
 *
 * The reference is vLLM's `vllm/models/inkling2/nvidia/dspark.py` and its speculator: a three-layer
 * dense transformer that denoises a block of `block_size` rows -- the anchor (the token the verify
 * just chose, at the committed end p) and block_size - 1 masks at p + 1 ... -- non-causally,
 * against context K and V it derived from the target's hidden states at every committed position.
 * Each mask row predicts the token at its own position (the reference's sample_from_anchor is
 * false), so the block is depth + 1 rows and the head reads rows 1..depth (RadDrafterDecl::block).
 *
 * What it reads of the target: four streams per committed token, `target_layer_ids` 5, 23, 41 and
 * 48 -- the running partial block entering layer applications 5, 23 and 41 (application numbers,
 * the loop unrolled to 48), and the final norm's output for 48 -- copied by the trunk into one
 * [rows, 4 * H] tap buffer, a token's four streams side by side.
 *
 *   context pass, per drafter layer i, over every row the trunk step ran:
 *       c    = rmsnorm(sum_t alpha[i][t] tap_t) * dflare_norm[i]       alpha = softmax(fuse_weight)
 *       K, V = k_proj_ctx c, v_proj_ctx c;  K = rmsnorm_head(K) * k_norm_ctx;  rope(K)
 *   query pass, per block:
 *       x    = [embed_norm(embed(anchor)), mask_embed x (block - 1)]
 *       per layer: q, k, v = q/k/v_proj(input_layernorm x); q, k per-head normed (q_norm, k_norm)
 *                  and rotated; attention over [context, block] (local layers: the 1024 positions
 *                  before the anchor and the block); x += o_proj; x += down(silu(gate) * up)
 *                  of post_attention_layernorm x
 *       h    = norm(x)
 *   head, per mask row: base logits from the target's lm_head over h, and the recurrent Markov walk
 *   (markov_rnn_greedy, docs/OPS.md) whose context is rnn_proj's [hidden | f_rnn] columns, where
 *   f_rnn = rmsnorm(sum_t alpha[3][t] tap_t at the last committed row) * markov_head.dflare_norm.
 *
 * ============================== WHAT IS DERIVED AT CONVERT ==============================
 *
 * `fuse_alpha` (softmax of fuse_weight) and the Markov head's `fold_weight` / `fold_bias` (the
 * target's head folded with hidden_out) are computed by the importer (core/format/checkpoint.cpp,
 * derive_inkling_dspark), as the reference computes them when it attaches the drafter. Without the
 * fold every draft position would read the 0.9 GB head instead of a [V, 256] table.
 *
 * ============================== PRECISION ==============================
 *
 * bf16 weights and activations around an f32 residual stream (the reference keeps a bf16 one).
 * The taps are mixed by `mix_rows` and rounded once before the norm, where the reference norms the
 * f32 mix; the rnn_proj context is two bf16 GEMMs widened to f32, where the reference keeps one f32
 * GEMM. Each moves a proposal, never the output: every proposal is verified by the target.
 */
#ifndef INKLING2_DSPARK_H
#define INKLING2_DSPARK_H

#include <arch/rad_arch.h>

#include <cstring>

namespace inkling2 {

using namespace rad::arch;

struct Dspark {
    enum { kTaps = 4, kMaxLayers = 4 };

    bool    on = false;
    Geom    g{};
    int64_t E = 0, QD = 0, KVD = 0, R = 0, block = 0, steps = 0, VR = 0, win = 0;
    double  cap = 0;
    int     base = 0;
    int     tap_app[kTaps] = {};         /* the applications whose input stream is tapped; -1: the
                                          * final norm's output */
    bool    local[kMaxLayers] = {};
    rad_kvgroup kv = 0;

    /* ---- buffers */
    rad_buf b_taps = 0;                  /* [max_tok, 4 E] bf16, written by the trunk */
    rad_buf b_cm = 0, b_cn = 0, b_ck = 0, b_ckn = 0, b_cv = 0;
    rad_buf b_e = 0, b_z = 0, b_h = 0, b_x = 0, b_d = 0, b_anc = 0;
    rad_buf b_q = 0, b_k = 0, b_v = 0, b_qn = 0, b_kn = 0, b_att = 0, b_gu = 0, b_ff = 0;
    rad_buf b_hd = 0, b_dlog = 0, b_dlog_sh = 0, b_chb = 0, b_ch = 0, b_fa = 0, b_fm = 0, b_fr = 0, b_cfb = 0;
    rad_buf b_cf = 0, b_scr = 0, b_tsc = 0, b_ttk = 0, b_tok = 0;

    /* ---- weights */
    rad_weight w_embed = 0, w_embed_norm = 0, w_head = 0;     /* the target's */
    rad_weight w_alpha[kMaxLayers] = {}, w_dflare[kMaxLayers] = {};
    rad_weight w_anorm[kMaxLayers] = {}, w_fnorm[kMaxLayers] = {};
    rad_weight w_q[kMaxLayers] = {}, w_k[kMaxLayers] = {}, w_v[kMaxLayers] = {}, w_o[kMaxLayers] = {};
    rad_weight w_qn[kMaxLayers] = {}, w_kn[kMaxLayers] = {};
    rad_weight w_kc[kMaxLayers] = {}, w_vc[kMaxLayers] = {}, w_kcn[kMaxLayers] = {};
    rad_weight w_gu[kMaxLayers] = {}, w_dn[kMaxLayers] = {};
    rad_weight w_mask = 0, w_norm = 0, w_mkdf = 0;
    rad_weight w_w1 = 0, w_wsp = 0, w_wh = 0, w_wf = 0, w_rb = 0, w_fold = 0, w_foldb = 0;

    /* ---- ops */
    rad_op op_tap = 0, op_tap_out = 0;   /* a stream (the trunk's, the final norm's) into its
                                          * column slice of the tap buffer */
    rad_op op_cmix[kMaxLayers] = {}, op_cdn[kMaxLayers] = {}, op_ck[kMaxLayers] = {};
    rad_op op_cv[kMaxLayers] = {}, op_ckn[kMaxLayers] = {};
    rad_op op_crope = 0, op_cstore = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_enorm = 0, op_zero = 0, op_mask = 0, op_agat = 0, op_asct = 0;
    rad_op op_widen = 0, op_widen2 = 0, op_add = 0, op_ar = 0;
    rad_op op_anorm[kMaxLayers] = {}, op_fnorm[kMaxLayers] = {};
    rad_op op_q[kMaxLayers] = {}, op_k[kMaxLayers] = {}, op_v[kMaxLayers] = {}, op_o[kMaxLayers] = {};
    rad_op op_qn[kMaxLayers] = {}, op_kn[kMaxLayers] = {};
    rad_op op_gu[kMaxLayers] = {}, op_dn[kMaxLayers] = {};
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn_local = 0, op_attn_full = 0, op_act = 0;
    rad_op op_outnorm = 0, op_hgat = 0, op_dlog = 0, op_dgather = 0, op_ch = 0, op_chw = 0;
    rad_op op_fgat = 0, op_fmix = 0, op_fnorm2 = 0, op_cf = 0, op_cfw = 0, op_markov = 0;

    int  declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const RadBuildCtx* ctx,
                 const Geom& target, int base_layer, int n_apps, rad_buf tap_src, rad_buf stream,
                 rad_buf final_out, rad_weight embed,
                 rad_weight embed_norm, rad_weight head, int64_t vocab_real, double softcap);
    /* The tap for the stream entering application a, or with a == -1 the final norm's output. */
    void tap(RadCtx* c, rad_buf src, int64_t T, int a) const;
    void context(RadCtx* c, const RadBatch* batch) const;
    void query(RadCtx* c, const RadBatch* batch) const;
};

/* Whether the container carries Inkling 2's drafter: the importer names it (add_draft). */
inline bool dspark_present(const RadModelMeta* meta) {
    const char* mt = rad_meta_gets(meta, "draft.model_type", "");
    return std::strcmp(mt, "inkling2_dspark") == 0;
}

/* The depth a container's drafter proposes: one per mask row. */
inline int64_t dspark_depth(const RadModelMeta* meta) {
    return rad_meta_geti(meta, "draft.block_size", 0) - 1;
}

inline int Dspark::declare(RadBuilder* b, Names& nm, const RadModelMeta* meta,
                           const RadBuildCtx* ctx, const Geom& target, int base_layer, int n_apps,
                           rad_buf tap_src, rad_buf stream, rad_buf final_out, rad_weight embed,
                           rad_weight embed_norm,
                           rad_weight head, int64_t vocab_real, double softcap) {
    RAD_ARCH_TRY(drafter_geom(g, target, meta, "Inkling 2 DSpark"));
    E = g.n_embd;
    QD = g.n_head * g.head_dim;
    KVD = g.n_head_kv * g.head_dim;
    R = rad_meta_geti(meta, "draft.markov_rank", 0);
    block = rad_meta_geti(meta, "draft.block_size", 0);
    steps = block - 1;
    VR = vocab_real;
    cap = softcap;
    base = base_layer;
    w_embed = embed;
    w_embed_norm = embed_norm;
    w_head = head;
    win = rad_meta_geti(meta, "draft.sliding_window_size", 0);
    const char* mh = rad_meta_gets(meta, "draft.markov_head_type", "");
    if (g.world > 1 && (g.n_vocab_all % g.world || (g.n_vocab * (int64_t)sizeof(float)) % 16)) {
        fprintf(stderr, "radiance: inkling2: the DSpark drafter at tp %d needs an evenly "
                        "sharded 16-byte-aligned vocabulary (%lld / %d = %lld)\n",
                g.world, (long long)g.n_vocab_all, g.world, (long long)g.n_vocab);
        return RAD_E_UNSUPPORTED;
    }
    if (std::strcmp(mh, "rnn_hidden") != 0 || R < 32 || R > 512 || R % 32) {
        fprintf(stderr, "radiance: inkling2: the drafter's Markov head is '%s' at rank %lld; this "
                        "plugin serves the release's recurrent head (rnn_hidden) at a rank that is "
                        "a multiple of 32 up to 512\n", mh, (long long)R);
        return RAD_E_UNSUPPORTED;
    }
    if (g.n_layer <= 0 || g.n_layer > kMaxLayers || block < 2 || win <= 0 ||
        rad_meta_geti(meta, "draft.hidden_size", E) != E) {
        fprintf(stderr, "radiance: inkling2: the drafter's config is not the release's shape "
                        "(%lld layers, block %lld, window %lld, hidden %lld against the target's "
                        "%lld)\n", (long long)g.n_layer, (long long)block, (long long)win,
                (long long)rad_meta_geti(meta, "draft.hidden_size", 0), (long long)E);
        return RAD_E_UNSUPPORTED;
    }
    if ((int64_t)ctx->max_spec != steps) {
        fprintf(stderr, "radiance: inkling2: the drafter denoises a block of %lld rows and proposes "
                        "%lld tokens, and this serve asks for %d; run it at %lld\n",
                (long long)block, (long long)steps, ctx->max_spec, (long long)steps);
        return RAD_E_UNSUPPORTED;
    }
    {
        int ids[8];
        const int n = parse_layer_ids(rad_meta_gets(meta, "draft.target_layer_ids", ""),
                                      n_apps + 1, ids, 8);
        bool ok = n == kTaps && ids[kTaps - 1] == n_apps;
        for (int t = 0; ok && t + 1 < kTaps; ++t) ok = ids[t] >= 0 && ids[t] < n_apps;
        if (!ok) {
            fprintf(stderr, "radiance: inkling2: the drafter's target_layer_ids are not %d layer "
                            "applications followed by %d (the final norm)\n", kTaps - 1, n_apps);
            return RAD_E_UNSUPPORTED;
        }
        for (int t = 0; t < kTaps; ++t) tap_app[t] = ids[t] == n_apps ? -1 : ids[t];
        for (int l = 0; l < kMaxLayers; ++l) local[l] = false;
        int lids[8];
        const int nl = parse_layer_ids(rad_meta_gets(meta, "draft.local_layer_ids", ""),
                                       g.n_layer, lids, 8);
        for (int i = 0; i < nl; ++i) local[lids[i]] = true;
    }
    b_taps = tap_src;

    /* One cache for the three layers, kept whole: the local layers bound their keys in the
     * attention, so the scheduler stages no window of its own (RadDrafterDecl::window 0). */
    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, g, 0, base, &kv));

    const int64_t T = g.max_tok, HD = g.head_dim;
    int64_t qr = g.max_seqs * block;
    if (qr > T) qr = T;
    const int64_t S = g.max_seqs, sr = S * steps;
    const int64_t V = g.n_vocab_all;

    /* ---- buffers */
    b_cm  = decl_b(b, nm.f("ds_cm"),  RAD_BF16, {T, E});
    b_cn  = decl_b(b, nm.f("ds_cn"),  RAD_BF16, {T, E});
    b_ck  = decl_b(b, nm.f("ds_ck"),  RAD_BF16, {T, KVD});
    b_ckn = decl_b(b, nm.f("ds_ckn"), RAD_BF16, {T, KVD});
    b_cv  = decl_b(b, nm.f("ds_cv"),  RAD_BF16, {T, KVD});
    b_e   = decl_b(b, nm.f("ds_e"),   RAD_BF16, {qr, E});
    b_z   = decl_b(b, nm.f("ds_z"),   RAD_BF16, {qr, E});
    b_h   = decl_b(b, nm.f("ds_h"),   RAD_BF16, {qr, E});
    b_x   = decl_b(b, nm.f("ds_x"),   RAD_F32,  {qr, E});
    b_d   = decl_b(b, nm.f("ds_d"),   RAD_F32,  {qr, E});
    b_anc = decl_b(b, nm.f("ds_anc"), RAD_BF16, {S, E});
    b_q   = decl_b(b, nm.f("ds_q"),   RAD_BF16, {qr, QD});
    b_k   = decl_b(b, nm.f("ds_k"),   RAD_BF16, {qr, KVD});
    b_v   = decl_b(b, nm.f("ds_v"),   RAD_BF16, {qr, KVD});
    b_qn  = decl_b(b, nm.f("ds_qn"),  RAD_BF16, {qr, QD});
    b_kn  = decl_b(b, nm.f("ds_kn"),  RAD_BF16, {qr, KVD});
    b_att = decl_b(b, nm.f("ds_att"), RAD_BF16, {qr, QD});
    b_gu  = decl_b(b, nm.f("ds_gu"),  RAD_BF16, {qr, 2 * g.n_ff});
    b_ff  = decl_b(b, nm.f("ds_ff"),  RAD_BF16, {qr, g.n_ff});
    b_hd  = decl_b(b, nm.f("ds_hd"),  RAD_BF16, {sr, E});
    b_dlog = decl_b(b, nm.f("ds_dlog"), RAD_F32, {sr, V});
    if (g.world > 1) {
        b_dlog_sh = decl_b(b, nm.f("ds_dlog_sh"), RAD_F32, {sr, g.n_vocab});
        if (!b_dlog_sh) return RAD_E_INVAL;
    }
    b_chb = decl_b(b, nm.f("ds_chb"), RAD_BF16, {sr, 3 * R});
    b_ch  = decl_b(b, nm.f("ds_ch"),  RAD_F32,  {sr, 3 * R});
    b_fa  = decl_b(b, nm.f("ds_fa"),  RAD_BF16, {S, kTaps * E});
    b_fm  = decl_b(b, nm.f("ds_fm"),  RAD_BF16, {S, E});
    b_fr  = decl_b(b, nm.f("ds_fr"),  RAD_BF16, {S, E});
    b_cfb = decl_b(b, nm.f("ds_cfb"), RAD_BF16, {S, 3 * R});
    b_cf  = decl_b(b, nm.f("ds_cf"),  RAD_F32,  {S, 3 * R});
    b_scr = decl_b(b, nm.f("ds_scr"), RAD_F32,  {S, 3 * R});
    b_tsc = decl_b(b, nm.f("ds_tsc"), RAD_F32,  {S, (VR + 63) / 64});
    b_ttk = decl_b(b, nm.f("ds_ttk"), RAD_I32,  {S, (VR + 63) / 64});
    /* The proposal buffer (RadDrafterDecl::proposal), read by the engine. */
    b_tok = decl_b(b, nm.f("ds_tokens"), RAD_I32, {S, steps});
    for (rad_buf h : { b_cm, b_cn, b_ck, b_ckn, b_cv, b_e, b_z, b_h, b_x, b_d, b_anc, b_q, b_k,
                       b_v, b_qn, b_kn, b_att, b_gu, b_ff, b_hd, b_dlog, b_chb, b_ch, b_fa, b_fm,
                       b_fr, b_cfb, b_cf, b_scr, b_tsc, b_ttk, b_tok })
        if (!h) return RAD_E_INVAL;

    /* ---- weights */
    auto one = [&](rad_weight* w, const char* name, const char* src, uint32_t dt,
                   std::initializer_list<int64_t> shape, RadWeightGroup grp,
                   int access = RAD_ACCESS_PER_TOKEN, int shard = RAD_SHARD_NONE) -> int {
        RAD_ARCH_TRY(map_copy(b, name, src));
        *w = decl_w(b, name, dt, shape, access, shard, grp);
        return *w ? RAD_OK : RAD_E_INVAL;
    };
    auto cols = [&](rad_weight* w, const char* name, const char* src, int64_t first, int64_t n,
                    int64_t rows) -> int {
        RadNameMap mp = map_base(name, RAD_MAP_COLS);
        mp.n_src = 1;
        mp.src[0] = src;
        mp.src_index[0] = (int)first;
        mp.src_rows[0] = n;
        RAD_ARCH_TRY(rad_decl_name_map(b, &mp));
        *w = decl_w(b, name, RAD_BF16, {rows, n}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        return *w ? RAD_OK : RAD_E_INVAL;
    };
    for (int i = 0; i < (int)g.n_layer + 1; ++i) {
        RAD_ARCH_TRY(map_rows(b, nm.f("ds.alpha.%d", i), { { nm.ckpt("draft.fuse_alpha"), i, 1 } }));
        w_alpha[i] = decl_w(b, nm.f("ds.alpha.%d", i), RAD_BF16, {1, kTaps}, RAD_ACCESS_PER_TOKEN,
                            RAD_SHARD_NONE, grp_model());
        if (!w_alpha[i]) return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(one(&w_mask, nm.f("ds.mask_embed"), nm.ckpt("draft.mask_embed"), RAD_BF16, {1, E},
                     grp_model()));
    RAD_ARCH_TRY(one(&w_norm, nm.f("ds.norm"), nm.ckpt("draft.norm.weight"), RAD_BF16, {E},
                     grp_model()));
    RAD_ARCH_TRY(one(&w_mkdf, nm.f("ds.mk.dflare"), nm.ckpt("draft.markov_head.dflare_norm.weight"),
                     RAD_BF16, {E}, grp_model()));
    RAD_ARCH_TRY(one(&w_w1, nm.f("ds.mk.w1"), nm.ckpt("draft.markov_head.markov_w1.weight"),
                     RAD_BF16, {V, R}, grp_model(), RAD_ACCESS_VOCAB));
    /* Brilliance: a container whose drafter was quantised (StillDeadcode's inkling-2-mini-iq8r:
     * int8 linears and an int8 folded head, walked by markov_rnn_greedy_q8) needs a later upstream
     * release. Say so by name, before the builder refuses the first int8 weight in its own words. */
    RAD_ARCH_TRY(map_copy(b, nm.f("ds.mk.fold"), nm.ckpt("draft.markov_head.fold_weight")));
    {
        RadEncoding fe{};
        if (weight_enc(b, nm.f("ds.mk.fold"), &fe) &&
            !(fe.n_planes == 1 && fe.plane[0].dtype == RAD_BF16)) {
            char en[256];
            rad_enc_format(&fe, en, sizeof en);
            fprintf(stderr, "radiance: inkling2: this container's DSpark drafter is quantised (its folded "
                            "head ds.mk.fold is %s); this build serves the bf16 drafter only (the int8 "
                            "drafter's walk, markov_rnn_greedy_q8, comes with a later upstream release). "
                            "Serve it without speculation: num-speculative-tokens 0\n", en);
            return RAD_E_UNSUPPORTED;
        }
    }
    w_fold = decl_w(b, nm.f("ds.mk.fold"), RAD_BF16, {V, R}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                    grp_model());
    if (!w_fold) return RAD_E_INVAL;
    RAD_ARCH_TRY(one(&w_foldb, nm.f("ds.mk.fold_b"), nm.ckpt("draft.markov_head.fold_bias"),
                     RAD_BF16, {V}, grp_model(), RAD_ACCESS_VOCAB));
    RAD_ARCH_TRY(one(&w_rb, nm.f("ds.mk.rnn_bias"), nm.ckpt("draft.markov_head.rnn_proj.bias"),
                     RAD_BF16, {3 * R}, grp_model()));
    /* rnn_proj's [state | w1[prev] | hidden | f_rnn] columns: the first two the walk applies a
     * step, the last two the plugin applies to every draft row at once. */
    {
        const char* rp = nm.ckpt("draft.markov_head.rnn_proj.weight");
        RAD_ARCH_TRY(cols(&w_wsp, nm.f("ds.mk.wsp"), rp, 0, 2 * R, 3 * R));
        RAD_ARCH_TRY(cols(&w_wh, nm.f("ds.mk.wh"), nm.ckpt("draft.markov_head.rnn_proj.weight"),
                          2 * R, E, 3 * R));
        RAD_ARCH_TRY(cols(&w_wf, nm.f("ds.mk.wf"), nm.ckpt("draft.markov_head.rnn_proj.weight"),
                          2 * R + E, E, 3 * R));
    }
    const int sh_row = g.world > 1 ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const int sh_col = g.world > 1 ? RAD_SHARD_COL : RAD_SHARD_NONE;
    for (int i = 0; i < (int)g.n_layer; ++i) {
        const RadWeightGroup grp = grp_layer(base + i);
        auto w = [&](rad_weight* out, const char* what, const char* from,
                     std::initializer_list<int64_t> shape, int shard = RAD_SHARD_NONE) {
            return one(out, nm.f("ds.%d.%s", i, what), nm.ckpt("draft.layers.%d.%s", i, from),
                       RAD_BF16, shape, grp, RAD_ACCESS_PER_TOKEN, shard);
        };
        RAD_ARCH_TRY(w(&w_dflare[i], "dflare", "dflare_norm.weight", {E}));
        RAD_ARCH_TRY(w(&w_anorm[i], "anorm", "input_layernorm.weight", {E}));
        RAD_ARCH_TRY(w(&w_fnorm[i], "fnorm", "post_attention_layernorm.weight", {E}));
        RAD_ARCH_TRY(w(&w_q[i], "q", "self_attn.q_proj.weight", {QD, E}, sh_row));
        RAD_ARCH_TRY(w(&w_k[i], "k", "self_attn.k_proj.weight", {KVD, E}, sh_row));
        RAD_ARCH_TRY(w(&w_v[i], "v", "self_attn.v_proj.weight", {KVD, E}, sh_row));
        RAD_ARCH_TRY(w(&w_o[i], "o", "self_attn.o_proj.weight", {E, QD}, sh_col));
        RAD_ARCH_TRY(w(&w_qn[i], "qn", "self_attn.q_norm.weight", {HD}));
        RAD_ARCH_TRY(w(&w_kn[i], "kn", "self_attn.k_norm.weight", {HD}));
        RAD_ARCH_TRY(w(&w_kc[i], "kc", "self_attn.k_proj_ctx.weight", {KVD, E}, sh_row));
        RAD_ARCH_TRY(w(&w_vc[i], "vc", "self_attn.v_proj_ctx.weight", {KVD, E}, sh_row));
        RAD_ARCH_TRY(w(&w_kcn[i], "kcn", "self_attn.k_norm_ctx.weight", {HD}));
        RAD_ARCH_TRY(w(&w_dn[i], "dn", "mlp.down_proj.weight", {E, g.n_ff}, sh_col));
        const char* gu = nm.f("ds.%d.gu", i);
        RAD_ARCH_TRY(map_concat(b, gu, { nm.ckpt("draft.layers.%d.mlp.gate_proj.weight", i),
                                         nm.ckpt("draft.layers.%d.mlp.up_proj.weight", i) }));
        w_gu[i] = decl_w(b, gu, RAD_BF16, {2 * g.n_ff, E}, RAD_ACCESS_PER_TOKEN, sh_row,
                         grp, 0, { g.n_ff, g.n_ff });
        if (!w_gu[i]) return RAD_E_INVAL;
    }

    /* ---- ops */
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N),
                                                RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(wt)), {x}, {y});
    };
    auto rms = [&](int64_t rows, int64_t n, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                                RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(wt)), {x}, {y});
    };
    auto mixr = [&](int64_t rows, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "mix_rows", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", E),
                                                 RAD_INT("k", kTaps), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(wt)), {x}, {y});
    };
    auto cast = [&](int64_t rows, int64_t n, const char* from, const char* to, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                             RAD_STR("from", from), RAD_STR("to", to)),
                           RAD_NOWEIGHTS), {x}, {y});
    };
    auto gather = [&](int64_t rows, int64_t n, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "gather_rows", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                    RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS), {x}, {y});
    };
    auto attn = [&](int64_t window) {
        return rw(b, RAD_OP(b, "attn_paged_anchor_window",
                            RAD_PARAMS(RAD_RANGE("q_len", 1, block), RAD_INT("head_dim", HD),
                                       RAD_INT("gqa", g.gqa()), RAD_INT("causal", 0),
                                       RAD_INT("window", window), RAD_INT("n_head", g.n_head),
                                       RAD_INT("block_size", RAD_KV_BLOCK),
                                       RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                       RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_NOWEIGHTS),
                  {b_qn}, {b_att});
    };

    /* The trunk's copies into the tap buffer: one column slice a stream. Declared here, after the
     * trunk, so the final norm's output lives until its copy. */
    op_tap = cast(T, E, "bf16", "bf16", stream, b_taps);
    op_tap_out = cast(T, E, "bf16", "bf16", final_out, b_taps);
    /* Both attentions before the kv_stores, which read the resolved block size back. The local
     * layers see the reference's window before the anchor, `sliding_window_size` positions
     * (attn_paged_anchor_window hides a key `window` or more behind the anchor); the full layer
     * sees every committed position, which a window past max_ctx says. */
    op_attn_local = attn(win + 1);
    op_attn_full = attn(g.max_ctx + 1);

    for (int i = 0; i < (int)g.n_layer; ++i) {
        op_cmix[i] = mixr(T, w_alpha[i], b_taps, b_cm);
        op_cdn[i] = rms(T, E, w_dflare[i], b_cm, b_cn);
        op_ck[i] = gemm(T, KVD, E, w_kc[i], b_cn, b_ck);
        op_cv[i] = gemm(T, KVD, E, w_vc[i], b_cn, b_cv);
        op_ckn[i] = rms(T * g.n_head_kv, HD, w_kcn[i], b_ck, b_ckn);
    }
    op_crope = rw(b, decl_rope(b, g, T, HD, g.n_head_kv, 0, HD, "neox"), {b_ckn}, {b_ckn});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", HD),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS), {b_ckn, b_cv}, {});

    /* the block: the anchor's normed embedding in row 0 of each sequence, mask_embed below it */
    op_embed = g.world > 1
        ? rw(b, RAD_OP(b, "embed_lookup", RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n_embd", E),
                                                    RAD_INT("n_vocab", g.n_vocab),
                                                    RAD_STR("dtype", "bf16"),
                                                    RAD_INT("vocab_offset", g.vocab_off)),
                       RAD_WEIGHTS(w_embed)), {}, {b_e})
        : rw(b, RAD_OP(b, "embed_lookup", RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n_embd", E),
                                                    RAD_INT("n_vocab", g.n_vocab),
                                                    RAD_STR("dtype", "bf16")),
                       RAD_WEIGHTS(w_embed)), {}, {b_e});
    if (g.world > 1) {
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                                   RAD_PARAMS(RAD_INT("world_size", g.world),
                                              RAD_RANGE("numel", E, qr * E),
                                              RAD_STR("dtype", "bf16"), RAD_INT("exact", 1)),
                                   RAD_NOWEIGHTS), {b_e}, {b_e});
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                             RAD_PARAMS(RAD_INT("world_size", g.world),
                                        RAD_RANGE("numel", E, qr * E),
                                        RAD_STR("dtype", "bf16"), RAD_INT("exact", 1)),
                             RAD_NOWEIGHTS), {b_h}, {b_h});
    }
    op_enorm = rms(qr, E, w_embed_norm, b_e, b_e);
    op_zero = rw(b, RAD_OP(b, "fill", RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n", E),
                                            RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                          RAD_NOWEIGHTS), {}, {b_z});
    op_mask = rw(b, RAD_OP(b, "table_add", RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n", E),
                                                 RAD_INT("L", 1), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w_mask)), {b_z}, {b_h});
    op_agat = gather(S, E, b_e, b_anc);
    op_asct = rw(b, RAD_OP(b, "scatter_rows", RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", E),
                                                    RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS), {b_anc, b_h}, {b_h});
    op_widen = cast(qr, E, "bf16", "f32", b_h, b_x);
    op_widen2 = cast(qr, E, "bf16", "f32", b_h, b_d);
    op_add = rw(b, RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n", E),
                                          RAD_STR("dtype", "f32")),
                         RAD_NOWEIGHTS), {b_x, b_d}, {b_x});
    for (int i = 0; i < (int)g.n_layer; ++i) {
        op_anorm[i] = rms(qr, E, w_anorm[i], b_x, b_h);
        op_q[i] = gemm(qr, QD, E, w_q[i], b_h, b_q);
        op_k[i] = gemm(qr, KVD, E, w_k[i], b_h, b_k);
        op_v[i] = gemm(qr, KVD, E, w_v[i], b_h, b_v);
        op_qn[i] = rms(qr * g.n_head, HD, w_qn[i], b_q, b_qn);
        op_kn[i] = rms(qr * g.n_head_kv, HD, w_kn[i], b_k, b_kn);
        op_o[i] = gemm(qr, E, QD, w_o[i], b_att, b_h);
        op_fnorm[i] = rms(qr, E, w_fnorm[i], b_x, b_h);
        op_gu[i] = gemm(qr, 2 * g.n_ff, E, w_gu[i], b_h, b_gu);
        op_dn[i] = gemm(qr, E, g.n_ff, w_dn[i], b_ff, b_h);
    }
    op_qrope = rw(b, decl_rope(b, g, qr, HD, g.n_head, 0, HD, "neox"), {b_qn}, {b_qn});
    op_krope = rw(b, decl_rope(b, g, qr, HD, g.n_head_kv, 0, HD, "neox"), {b_kn}, {b_kn});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("head_dim", HD),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS), {b_kn, b_v}, {});
    op_act = rw(b, RAD_OP(b, "act_mul", RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n", g.n_ff),
                                              RAD_STR("act", "silu"), RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS), {b_gu}, {b_ff});
    op_outnorm = rms(qr, E, w_norm, b_x, b_h);

    /* the head: the mask rows' base logits and rnn_proj's hidden columns, the anchor's f_rnn and
     * its columns, then the walk */
    op_hgat = gather(sr, E, b_h, b_hd);
    if (g.world > 1) {
        op_dlog = rw(b, RAD_OP(b, "logits_gemm", RAD_PARAMS(RAD_RANGE("M", 1, sr), RAD_INT("n_vocab", g.n_vocab),
                                                       RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_head)), {b_hd}, {b_dlog_sh});
        op_dgather = rw(b, RAD_OP(b, "all_gather", RAD_PARAMS(RAD_INT("world_size", g.world),
                                                          RAD_INT("numel", sr * g.n_vocab),
                                                          RAD_STR("dtype", "f32"),
                                                          RAD_INT("row", g.n_vocab)),
                                  RAD_NOWEIGHTS), {b_dlog_sh}, {b_dlog});
        if (!op_embed_ar || !op_ar || !op_dgather) return RAD_E_UNSUPPORTED;
    } else {
        op_dlog = rw(b, RAD_OP(b, "logits_gemm", RAD_PARAMS(RAD_RANGE("M", 1, sr), RAD_INT("n_vocab", g.n_vocab),
                                                       RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_head)), {b_hd}, {b_dlog});
    }
    op_ch = gemm(sr, 3 * R, E, w_wh, b_hd, b_chb);
    op_chw = cast(sr, 3 * R, "bf16", "f32", b_chb, b_ch);
    op_fgat = gather(S, kTaps * E, b_taps, b_fa);
    op_fmix = mixr(S, w_alpha[g.n_layer], b_fa, b_fm);
    op_fnorm2 = rms(S, E, w_mkdf, b_fm, b_fr);
    op_cf = gemm(S, 3 * R, E, w_wf, b_fr, b_cfb);
    op_cfw = cast(S, 3 * R, "bf16", "f32", b_cfb, b_cf);
    op_markov = rw(b, RAD_OP(b, "markov_rnn_greedy",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("steps", steps),
                                        RAD_INT("rank", R), RAD_INT("n_vocab", VR),
                                        RAD_INT("anchor_stride", block), RAD_F64("cap", cap),
                                        RAD_INT("vocab_rows", V), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_w1, w_wsp, w_rb, w_fold, w_foldb)),
                   {b_dlog, b_ch, b_cf}, {b_scr, b_tsc, b_ttk, b_tok});

    const rad_op all[] = { op_tap, op_tap_out, op_attn_local, op_attn_full, op_crope, op_cstore, op_embed,
                           op_enorm, op_zero, op_mask, op_agat, op_asct, op_widen, op_widen2,
                           op_add, op_qrope, op_krope, op_store, op_act, op_outnorm, op_hgat,
                           op_dlog, op_ch, op_chw, op_fgat, op_fmix, op_fnorm2, op_cf, op_cfw,
                           op_markov };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; ++i)
        if (!all[i]) {
            fprintf(stderr, "radiance: inkling2: the DSpark drafter's op %zu (in declaration order) "
                            "was refused\n", i);
            return RAD_E_INVAL;
        }
    for (int i = 0; i < (int)g.n_layer; ++i)
        if (!op_cmix[i] || !op_cdn[i] || !op_ck[i] || !op_cv[i] || !op_ckn[i] || !op_anorm[i] ||
            !op_q[i] || !op_k[i] || !op_v[i] || !op_qn[i] || !op_kn[i] || !op_o[i] ||
            !op_fnorm[i] || !op_gu[i] || !op_dn[i]) {
            fprintf(stderr, "radiance: inkling2: the DSpark drafter's layer %d declared an op with "
                            "no kernel\n", i);
            return RAD_E_INVAL;
        }

    RadDrafterDecl drf{};
    drf.name           = "dspark";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = block;
    drf.proposal       = b_tok;
    drf.proposal_pitch = steps;
    drf.window         = 0;
    /* The mask rows' token is never read: they embed mask_embed. Any id in the vocabulary. */
    drf.mask_token     = 0;
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    on = true;
    rad_note(b, "Inkling 2 DSpark drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x "
                "%lld, ffn %lld, local window %lld, a block of %lld rows proposes %lld tokens, the "
                "rank-%lld recurrent Markov head over %lld tokens; taps at applications %d, %d, %d "
                "and the final norm",
             (long long)g.n_layer, base, base + (int)g.n_layer - 1, (long long)g.n_head,
             (long long)g.n_head_kv, (long long)HD, (long long)g.n_ff, (long long)win,
             (long long)block, (long long)steps, (long long)R, (long long)VR, tap_app[0], tap_app[1],
             tap_app[2]);
    return RAD_OK;
}

inline void Dspark::tap(RadCtx* c, rad_buf src, int64_t T, int a) const {
    if (!on) return;
    for (int t = 0; t < kTaps; ++t)
        if (tap_app[t] == a)
            RAD_ISSUE_N(c, a < 0 ? op_tap_out : op_tap, T, brows(src, T),
                        bcol(b_taps, (int64_t)t * E, E, T));
}

/* One row a committed token, over the trunk step's rows in order. */
inline void Dspark::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
    const RadOperand taps = RAD_P_T2(rad_buf_ptr(c, b_taps), RAD_BF16, T * kTaps, E);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    for (int i = 0; i < (int)g.n_layer; ++i) {
        RAD_ISSUE_N(c, op_cmix[i], T, taps, RAD_W(w_alpha[i]), brows(b_cm, T));
        RAD_ISSUE_N(c, op_cdn[i], T, brows(b_cm, T), RAD_W(w_dflare[i]), brows(b_cn, T));
        RAD_ISSUE_N(c, op_ck[i], T, brows(b_cn, T), RAD_W(w_kc[i]), brows(b_ck, T));
        RAD_ISSUE_N(c, op_cv[i], T, brows(b_cn, T), RAD_W(w_vc[i]), brows(b_cv, T));
        RAD_ISSUE_N(c, op_ckn[i], T * g.n_head_kv, brows(b_ck, T), RAD_W(w_kcn[i]), brows(b_ckn, T));
        RAD_ISSUE_N(c, op_crope, T, brows(b_ckn, T), pos);
        RAD_ISSUE_N(c, op_cstore, T, brows(b_ckn, T), brows(b_cv, T), praw(slot, RAD_I32, T),
                    kv_cache(kv, base + i));
    }
}

/* `block` rows a sequence: the drafter's layers, the head over the mask rows (out_ids) and the
 * walk. The sequences' blocks are laid end to end, so cu_seqlens' first n_seq entries are the
 * anchor rows and the anchor of sequence s is token_ids[s * block]. */
inline void Dspark::query(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok, S = batch->n_seq;
    if (T <= 0 || S <= 0) return;
    if (!batch->anchor_rows || T != S * block) {
        rad_step_fail(c, "inkling2: the DSpark query pass is not one block a sequence with its "
                         "anchor rows");
        return;
    }
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S);

    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_embed), brows(b_e, T));
    if (op_embed_ar) RAD_ISSUE_N(c, op_embed_ar, T * E, brows(b_e, T), RAD_NONE);
    RAD_ISSUE_N(c, op_enorm, T, brows(b_e, T), RAD_W(w_embed_norm), brows(b_e, T));
    RAD_ISSUE_N(c, op_zero, T, brows(b_z, T));
    RAD_ISSUE_N(c, op_mask, T, brows(b_z, T), RAD_W(w_mask), brows(b_h, T));
    RAD_ISSUE_N(c, op_agat, S, brows(b_e, T), cu, brows(b_anc, S));
    RAD_ISSUE_N(c, op_asct, S, brows(b_anc, S), cu, brows(b_h, T));
    RAD_ISSUE_N(c, op_widen, T, brows(b_h, T), brows(b_x, T));

    for (int i = 0; i < (int)g.n_layer; ++i) {
        RAD_ISSUE_N(c, op_anorm[i], T, brows(b_x, T), RAD_W(w_anorm[i]), brows(b_h, T));
        RAD_ISSUE_N(c, op_q[i], T, brows(b_h, T), RAD_W(w_q[i]), brows(b_q, T));
        RAD_ISSUE_N(c, op_k[i], T, brows(b_h, T), RAD_W(w_k[i]), brows(b_k, T));
        RAD_ISSUE_N(c, op_v[i], T, brows(b_h, T), RAD_W(w_v[i]), brows(b_v, T));
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(b_q, T), RAD_W(w_qn[i]), brows(b_qn, T));
        RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv, brows(b_k, T), RAD_W(w_kn[i]), brows(b_kn, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(b_qn, T), pos);
        RAD_ISSUE_N(c, op_krope, T, brows(b_kn, T), pos);
        /* The block's own K and V past the committed end, where every row of the block sees them;
         * the next context pass overwrites them with the committed tokens'. */
        RAD_ISSUE_N(c, op_store, T, brows(b_kn, T), brows(b_v, T), praw(slot, RAD_I32, T),
                    kv_cache(kv, base + i));
        RAD_ISSUE_N(c, local[i] ? op_attn_local : op_attn_full, batch->max_q_len, brows(b_qn, T),
                    kv_cache(kv, base + i),
                    praw2(kvb ? kvb->block_table : nullptr, RAD_I32, S, kvb ? kvb->block_table_pitch : 0),
                    praw(kvb ? kvb->seqused : nullptr, RAD_I32, S), RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, S + 1), brows(b_att, T));
        RAD_ISSUE_N(c, op_o[i], T, brows(b_att, T), RAD_W(w_o[i]), brows(b_h, T));
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * E, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_widen2, T, brows(b_h, T), brows(b_d, T));
        RAD_ISSUE_N(c, op_add, T, brows(b_x, T), brows(b_d, T), brows(b_x, T));
        RAD_ISSUE_N(c, op_fnorm[i], T, brows(b_x, T), RAD_W(w_fnorm[i]), brows(b_h, T));
        RAD_ISSUE_N(c, op_gu[i], T, brows(b_h, T), RAD_W(w_gu[i]), brows(b_gu, T));
        RAD_ISSUE_N(c, op_act, T, bcol(b_gu, 0, g.n_ff, T), bcol(b_gu, g.n_ff, g.n_ff, T),
                    brows(b_ff, T));
        RAD_ISSUE_N(c, op_dn[i], T, brows(b_ff, T), RAD_W(w_dn[i]), brows(b_h, T));
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * E, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_widen2, T, brows(b_h, T), brows(b_d, T));
        RAD_ISSUE_N(c, op_add, T, brows(b_x, T), brows(b_d, T), brows(b_x, T));
    }
    RAD_ISSUE_N(c, op_outnorm, T, brows(b_x, T), RAD_W(w_norm), brows(b_h, T));

    const int64_t N = batch->n_out;
    if (N != S * steps) {
        rad_step_fail(c, "inkling2: the DSpark query pass names a head row count other than the "
                         "mask rows'");
        return;
    }
    RAD_ISSUE_N(c, op_hgat, N, brows(b_h, T), praw(batch->out_ids, RAD_I32, N), brows(b_hd, N));
    if (op_dgather) {
        RAD_ISSUE_N(c, op_dlog, N, brows(b_hd, N), RAD_W(w_head), brows(b_dlog_sh, N));
        RAD_ISSUE(c, op_dgather, brows(b_dlog_sh, N), brows(b_dlog, N));
    } else {
        RAD_ISSUE_N(c, op_dlog, N, brows(b_hd, N), RAD_W(w_head), brows(b_dlog, N));
    }
    RAD_ISSUE_N(c, op_ch, N, brows(b_hd, N), RAD_W(w_wh), brows(b_chb, N));
    RAD_ISSUE_N(c, op_chw, N, brows(b_chb, N), brows(b_ch, N));
    /* f_rnn: the four streams at each sequence's last committed row of the context pass */
    RAD_ISSUE_N(c, op_fgat, S, RAD_B(b_taps), praw(batch->anchor_rows, RAD_I32, S), brows(b_fa, S));
    RAD_ISSUE_N(c, op_fmix, S, RAD_P_T2(rad_buf_ptr(c, b_fa), RAD_BF16, S * kTaps, E),
                RAD_W(w_alpha[g.n_layer]), brows(b_fm, S));
    RAD_ISSUE_N(c, op_fnorm2, S, brows(b_fm, S), RAD_W(w_mkdf), brows(b_fr, S));
    RAD_ISSUE_N(c, op_cf, S, brows(b_fr, S), RAD_W(w_wf), brows(b_cfb, S));
    RAD_ISSUE_N(c, op_cfw, S, brows(b_cfb, S), brows(b_cf, S));
    RAD_ISSUE_N(c, op_markov, S, brows(b_dlog, N), brows(b_ch, N), brows(b_cf, S),
                praw(batch->token_ids, RAD_I32, T), RAD_W(w_w1), RAD_W(w_wsp), RAD_W(w_rb),
                RAD_W(w_fold), RAD_W(w_foldb), brows(b_scr, S), brows(b_tsc, S), brows(b_ttk, S),
                brows(b_tok, S));
}

}  /* namespace inkling2 */

#endif
