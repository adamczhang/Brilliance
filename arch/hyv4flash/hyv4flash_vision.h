/* hyv4flash_vision.h -- Hy4-flash's vision tower and projector (transformers c2ec8505's
 * HYV4FlashVisionTransformer and HYV4FlashVisionPatchMerger), served on an encoder pass
 * (RadBatch::enc). Per pass of P patches in S segments (a picture, or a video frame):
 *
 *   x  = patches @ Wp^T + bp                  the Conv2d patch embedding as a linear layer over a
 *                                             patch row (channel, y, x); temporal depth 1
 *   x += pos_table[1:] resampled to each segment's grid   grid_embed_sf: torch's half-pixel
 *                                             bilinear at scale_factor (size + 0.1) / side, the
 *                                             table's row 0 (a class token the tower never reads)
 *                                             dropped at convert; rounded before the add
 *   depth times:
 *     x += o(attn(q, k, v of LN1(x)))         biased q, k, v (fused at convert) and o; full
 *                                             attention within a segment, no rotary
 *     x += fc2(gelu(fc1(LN2(x))))             exact GELU, biases
 *   m  = gelu(conv2x2(RMS(x)) + b0)           the merger: `before_rms` (a gain), then the 2 x 2
 *                                             stride-2 convolution to 4 x width (conv2d_unfold
 *                                             and a GEMM over the weight as stored)
 *   r  = m @ W2^T + b2                        the 1 x 1 convolution to the text width
 *   out = each merged row of r followed by `image_newline`
 *
 * The processor (core/mm, Family::HyV4Flash) stages patches in merge-block order, so a block's
 * four patches are consecutive rows and the merger's input is a view; everything else in the tower
 * is per patch or attends within a segment, and the position table reads each patch's own
 * coordinates. A segment of hp x wp patches gives (hp / 2) x (wp / 2 + 1) output rows: its merged
 * grid with one newline row closing each merged row, the reference's layout and the rotary grid
 * the core lays out. The newline rows are written by copies (the projected rows pitched into the
 * layout, the newline cells from a column filled from the weight), as unlimitedocr's layout is.
 *
 * Every rank runs the whole tower with replicated, per-request weights (the core reads rank 0's
 * rows). The weights stay bf16 (a recipe rule over them is refused by name). */
#ifndef HYV4FLASH_VISION_H
#define HYV4FLASH_VISION_H

#include "hyv4flash_common.h"

#include <rad_device.h>

namespace hyv4flash {

struct Vision {
    int64_t depth = 0, width = 0, heads = 0, head_dim = 0, ff = 0, patch = 0, chans = 0;
    int64_t merge = 0, side = 0, n_embd = 0, patch_dim = 0, P = 0, R = 0, Rout = 0;
    double eps = 1e-5, offset = 0.1;
    bool on = false;

    struct Layer {
        rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0, qkv_w = 0, qkv_b = 0, o_w = 0, o_b = 0;
        rad_weight up_w = 0, up_b = 0, down_w = 0, down_b = 0;
        rad_op ln1 = 0, qkv = 0, attn = 0, proj = 0, ln2 = 0, up = 0, down = 0;
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, b_patch = 0, w_pos = 0, w_rms = 0, w_m0 = 0, b_m0 = 0, w_m2 = 0,
               b_m2 = 0, w_nl = 0;
    rad_op op_patch = 0, op_pos = 0, op_rms = 0, op_unfold = 0, op_m0 = 0, op_m2 = 0;
    rad_op op_nl_ids = 0, op_nl = 0;
    rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_ff = 0, b_post = 0, b_unf = 0, b_m = 0,
            b_r = 0, b_nl = 0, b_out = 0, b_zero = 0;
    /* debug taps (RADIANCE_HY4_TAPS): the embedded patches, each block's output, the merger's
     * rows before the layout; global keys, a pass's rows. */
    Tap t_embed{}, t_merged{};
    std::vector<Tap> t_blk;

    /* `g0` the first movement group (the drafter's follow the trunk's). `serve` false declares
     * the weights alone, in their served form (a container from a config that states too little
     * to run the tower still carries it). */
    int declare(RadBuilder* b, Names& nm, const Shape& s, const RadBuildCtx* ctx, int g0,
                bool serve) {
        on = false;
        depth = s.v_layers;
        width = s.v_embd;
        heads = s.v_heads;
        ff = s.v_ff;
        patch = s.v_patch;
        chans = s.v_chan;
        merge = s.v_merge;
        n_embd = s.n_embd;
        eps = s.v_eps;
        side = s.v_side;
        if (serve && (heads <= 0 || width % heads)) return RAD_E_INVAL;
        if (depth <= 0 || width <= 0 || ff <= 0 || patch <= 0 || chans != 3 || merge <= 0 ||
            side <= 0 || s.v_pos != side * side + 1) {
            fprintf(stderr, "radiance: hyv4flash: the vision_config is incomplete or inconsistent "
                            "(layers %lld, hidden %lld, heads %lld, ff %lld, patch %lld, channels "
                            "%lld, merge %lld; a %lld x %lld position grid against %lld table "
                            "rows)\n",
                    (long long)depth, (long long)width, (long long)heads, (long long)ff,
                    (long long)patch, (long long)chans, (long long)merge, (long long)side,
                    (long long)side, (long long)s.v_pos);
            return RAD_E_INVAL;
        }
        head_dim = heads > 0 ? width / heads : 0;
        patch_dim = chans * patch * patch;
        const int64_t mm = merge * merge;
        P = ctx->max_enc_patches;
        if (ctx->shape_probe) {
            const int64_t r = std::min<int64_t>(P, ctx->max_tok);
            P = std::max<int64_t>(mm, r / mm * mm);
        }
        if (P <= 0 || P % mm) return RAD_E_SHAPE;
        R = P / mm;
        /* A segment's newline rows are its merged rows' count over its merged width, so never
         * more than its merged rows: a pass leaves at most 2 R rows. */
        Rout = 2 * R;
        const int ACC = RAD_ACCESS_PER_REQUEST;

        /* bf16 as the checkpoint ships every tower weight; a recipe that touched one is
         * refused (nothing here reads quantised codes). */
        auto plain = [&](const char* dn) {
            RadEncoding e{};
            if (weight_enc(b, dn, &e) &&
                !(rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16)) {
                fprintf(stderr, "radiance: hyv4flash: the vision tower's '%s' is not plain bf16; "
                                "the tower is served as the checkpoint ships it\n", dn);
                return false;
            }
            return true;
        };
        auto bw = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                      int grp) -> rad_weight {
            if (map_copy(b, dn, src) < 0 || !plain(dn)) return 0;
            return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, grp_layer(grp));
        };
        /* [width, C, p, p] read as [width, C * p * p]: a patch row is (channel, y, x). */
        w_patch = bw(nm.f("v.patch.weight"), "vit.embeddings.patch_embedding.weight",
                     { width, patch_dim }, g0);
        b_patch = bw(nm.f("v.patch.bias"), "vit.embeddings.patch_embedding.bias", { width }, g0);
        /* Rows 1 .. side^2 of the table: row 0 is the class token's. */
        {
            const char* dn = nm.f("v.position.weight");
            if (map_rows(b, dn, { { "vit.embeddings.position_embedding.weight", 1,
                                    side * side } }) < 0 || !plain(dn))
                return RAD_E_INVAL;
            w_pos = decl_w(b, dn, RAD_BF16, { side * side, width }, ACC, RAD_SHARD_NONE,
                           grp_layer(g0));
        }
        if (!w_patch || !b_patch || !w_pos) return RAD_E_INVAL;
        L.assign((size_t)depth, Layer{});
        for (int64_t i = 0; i < depth; ++i) {
            Layer& l = L[(size_t)i];
            const int li = (int)i, grp = g0 + 1 + li;
            const std::string vp = "vit.layers." + std::to_string(li) + ".";
            auto src = [&](const char* t) { return nm.ckpt("%s%s", vp.c_str(), t); };
            l.n1w = bw(nm.f("v.blk.%d.attn_norm.weight", li), src("input_layernorm.weight"),
                       { width }, grp);
            l.n1b = bw(nm.f("v.blk.%d.attn_norm.bias", li), src("input_layernorm.bias"),
                       { width }, grp);
            const char* qw = nm.f("v.blk.%d.attn_qkv.weight", li);
            const char* qb = nm.f("v.blk.%d.attn_qkv.bias", li);
            if (map_concat(b, qw, { src("self_attn.q_proj.weight"), src("self_attn.k_proj.weight"),
                                    src("self_attn.v_proj.weight") }) < 0 ||
                map_concat(b, qb, { src("self_attn.q_proj.bias"), src("self_attn.k_proj.bias"),
                                    src("self_attn.v_proj.bias") }) < 0 ||
                !plain(qw) || !plain(qb))
                return RAD_E_INVAL;
            l.qkv_w = decl_w(b, qw, RAD_BF16, { 3 * width, width }, ACC, RAD_SHARD_NONE,
                             grp_layer(grp));
            l.qkv_b = decl_w(b, qb, RAD_BF16, { 3 * width }, ACC, RAD_SHARD_NONE, grp_layer(grp));
            l.o_w = bw(nm.f("v.blk.%d.attn_output.weight", li), src("self_attn.o_proj.weight"),
                       { width, width }, grp);
            l.o_b = bw(nm.f("v.blk.%d.attn_output.bias", li), src("self_attn.o_proj.bias"),
                       { width }, grp);
            l.n2w = bw(nm.f("v.blk.%d.ffn_norm.weight", li), src("post_attention_layernorm.weight"),
                       { width }, grp);
            l.n2b = bw(nm.f("v.blk.%d.ffn_norm.bias", li), src("post_attention_layernorm.bias"),
                       { width }, grp);
            l.up_w = bw(nm.f("v.blk.%d.ffn_up.weight", li), src("mlp.dense_h_to_4h.weight"),
                        { ff, width }, grp);
            l.up_b = bw(nm.f("v.blk.%d.ffn_up.bias", li), src("mlp.dense_h_to_4h.bias"), { ff },
                        grp);
            l.down_w = bw(nm.f("v.blk.%d.ffn_down.weight", li), src("mlp.dense_4h_to_h.weight"),
                          { width, ff }, grp);
            l.down_b = bw(nm.f("v.blk.%d.ffn_down.bias", li), src("mlp.dense_4h_to_h.bias"),
                          { width }, grp);
            if (!l.n1w || !l.n1b || !l.qkv_w || !l.qkv_b || !l.o_w || !l.o_b || !l.n2w ||
                !l.n2b || !l.up_w || !l.up_b || !l.down_w || !l.down_b)
                return RAD_E_INVAL;
        }
        const int gm = g0 + 1 + (int)depth;
        const int64_t C = mm * width;
        w_rms = bw(nm.f("v.merger.norm.weight"), "vit.perceive.norm.weight", { width }, gm);
        /* [C, width, m, m] read as [C, width * m * m]: conv2d_unfold's column order. */
        w_m0 = bw(nm.f("v.merger.merge.weight"), "vit.perceive.proj.0.weight", { C, C }, gm);
        b_m0 = bw(nm.f("v.merger.merge.bias"), "vit.perceive.proj.0.bias", { C }, gm);
        w_m2 = bw(nm.f("v.merger.proj.weight"), "vit.perceive.proj.2.weight", { n_embd, C }, gm);
        b_m2 = bw(nm.f("v.merger.proj.bias"), "vit.perceive.proj.2.bias", { n_embd }, gm);
        /* The newline row, a one-row table: embed_lookup copies it into the layout's column. */
        w_nl = bw(nm.f("v.merger.newline"), "vit.perceive.image_newline", { 1, n_embd }, gm);
        if (!w_rms || !w_m0 || !b_m0 || !w_m2 || !b_m2 || !w_nl) return RAD_E_INVAL;
        if (!serve) return RAD_OK;

        /* ---- activations. `post` is the merger norm's output declared [P / m, m * width]: the
         * norm writes P rows of width and conv2d_unfold reads the same bytes as P / m kernel
         * rows of m patches. */
        b_x    = decl_b(b, nm.f("v.x"),    RAD_BF16, { P, width });
        b_h    = decl_b(b, nm.f("v.h"),    RAD_BF16, { P, width });
        b_qkv  = decl_b(b, nm.f("v.qkv"),  RAD_BF16, { P, 3 * width });
        b_att  = decl_b(b, nm.f("v.att"),  RAD_BF16, { P, width });
        b_ff   = decl_b(b, nm.f("v.ff"),   RAD_BF16, { P, ff });
        b_post = decl_b(b, nm.f("v.post"), RAD_BF16, { P / merge, merge * width });
        b_unf  = decl_b(b, nm.f("v.unf"),  RAD_BF16, { R, C });
        b_m    = decl_b(b, nm.f("v.m"),    RAD_BF16, { R, C });
        b_r    = decl_b(b, nm.f("v.r"),    RAD_BF16, { R, n_embd });
        b_nl   = decl_b(b, nm.f("v.nl"),   RAD_BF16, { R, n_embd });
        b_out  = decl_b(b, nm.f("v.out"),  RAD_BF16, { Rout, n_embd });
        b_zero = decl_b(b, nm.f("v.zero"), RAD_I32, { R });
        if (!b_x || !b_h || !b_qkv || !b_att || !b_ff || !b_post || !b_unf || !b_m || !b_r ||
            !b_nl || !b_out || !b_zero)
            return RAD_E_INVAL;

        auto gemm = [&](int64_t rows, int64_t N, int64_t K, const char* act, rad_weight w,
                        rad_weight bias) -> rad_op {
            return RAD_OP(b, "gemm_nt_bias",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_STR("dtype", "bf16"), RAD_STR("act", act)),
                          RAD_WEIGHTS(w, bias));
        };
        auto lnorm = [&](rad_weight w, rad_weight bias) -> rad_op {
            return RAD_OP(b, "layernorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_F64("eps", eps),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w, bias));
        };
        /* The longest segment attention sees: a picture is one segment of one pass. */
        const int64_t max_seg = P;
        op_patch = rw(b, gemm(P, width, patch_dim, "none", w_patch, b_patch), {}, { b_x });
        op_pos = rw(b, RAD_OP(b, "grid_embed_sf",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                         RAD_INT("side", side), RAD_F64("offset", offset),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_pos)),
                    { b_x }, { b_x });
        HY4_NEED(op_patch && op_pos, "the tower's patch embedding (gemm_nt_bias, grid_embed_sf)");
        RAD_ARCH_TRY(t_embed.declare(b, nm, P, "v_embed", -1, RAD_BF16, width, b_x));
        t_blk.assign((size_t)depth, Tap{});
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            l.ln1  = rw(b, lnorm(l.n1w, l.n1b), { b_x }, { b_h });
            l.qkv  = rw(b, gemm(P, 3 * width, width, "none", l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
            l.attn = rw(b, RAD_OP(b, "attn_dense",
                                  RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                             RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                             RAD_STR("dtype", "bf16"),
                                             RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                             RAD_INT("max_seqlen", max_seg)),
                                  RAD_NOWEIGHTS),
                        { b_qkv }, { b_att });
            l.proj = rw(b, gemm(P, width, width, "none", l.o_w, l.o_b), { b_att, b_x }, { b_x });
            l.ln2  = rw(b, lnorm(l.n2w, l.n2b), { b_x }, { b_h });
            l.up   = rw(b, gemm(P, ff, width, "gelu", l.up_w, l.up_b), { b_h }, { b_ff });
            l.down = rw(b, gemm(P, width, ff, "none", l.down_w, l.down_b), { b_ff, b_x }, { b_x });
            HY4_NEED(l.ln1 && l.qkv && l.attn && l.proj && l.ln2 && l.up && l.down,
                     "a tower block's ops (layernorm, gemm_nt_bias, attn_dense)");
            RAD_ARCH_TRY(t_blk[i].declare(b, nm, P, "v_block", (int)i, RAD_BF16, width, b_x));
        }
        op_rms = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                         RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_rms)),
                    { b_x }, { b_post });
        /* Input rows are P / m (one kernel row of m patches each) and a segment's rows its
         * patches over m, so a segment of whole merged blocks starts on a whole block. */
        op_unfold = rw(b, RAD_OP(b, "conv2d_unfold",
                                 RAD_PARAMS(RAD_RANGE("M", 1, P / merge), RAD_INT("width", merge),
                                            RAD_INT("cin", width), RAD_INT("kh", merge),
                                            RAD_INT("kw", merge), RAD_INT("stride", merge),
                                            RAD_INT("pad", 0), RAD_INT("cu_div", merge),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { b_post }, { b_unf });
        op_m0 = rw(b, gemm(R, C, C, "gelu", w_m0, b_m0), { b_unf }, { b_m });
        op_nl_ids = rw(b, RAD_OP(b, "iota",
                                 RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("start", 0),
                                            RAD_INT("step", 0)),
                                 RAD_NOWEIGHTS),
                       {}, { b_zero });
        op_nl = rw(b, RAD_OP(b, "embed_lookup",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n_embd", n_embd),
                                        RAD_INT("n_vocab", 1), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_nl)),
                   { b_zero }, { b_nl });
        /* The layout's copies (step) read b_r and b_nl and write b_out, outside any op, right
         * behind this GEMM, the pass's last op. Stating them as this op's reads and writes keeps
         * the three live together, so the planner gives them distinct bytes. */
        op_m2 = rw(b, gemm(R, n_embd, C, "none", w_m2, b_m2), { b_m, b_nl }, { b_r, b_out });
        HY4_NEED(op_rms && op_unfold && op_m0 && op_m2 && op_nl_ids && op_nl,
                 "the tower's merger (rmsnorm, conv2d_unfold, gemm_nt_bias, iota, embed_lookup)");
        RAD_ARCH_TRY(t_merged.declare(b, nm, R, "v_merged", -1, RAD_BF16, n_embd, b_r));

        if (!ctx->shape_probe) {
            RadEncoderDecl e{};
            e.name        = "vision";
            e.modalities  = RAD_MM_IMAGE | RAD_MM_VIDEO;
            e.patch_dim   = patch_dim;
            e.merge       = mm;
            e.max_patches = P;
            e.n_embd      = n_embd;
            e.out         = b_out;
            RAD_ARCH_TRY(rad_declare_encoder(b, &e));
            rad_note(b, "Hy4-flash tower: %lld blocks of %lld (%lld heads x %lld), ff %lld, patch "
                        "%lld, a %lld x %lld position grid; merger %lld -> %lld -> %lld with a "
                        "newline a merged row; %lld patches a pass",
                     (long long)depth, (long long)width, (long long)heads, (long long)head_dim,
                     (long long)ff, (long long)patch, (long long)side, (long long)side,
                     (long long)(mm * width), (long long)(mm * width), (long long)n_embd,
                     (long long)P);
        }
        on = true;
        return RAD_OK;
    }

    /* One encoder pass. The layout needs the segments' grids on the host (RadBatch::enc_seg_h,
     * enc_seg_w, in patches). */
    bool step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
        const int64_t mm = merge * merge, Rn = Pn / mm;
        if (!on || !(batch->enc & (RAD_MM_IMAGE | RAD_MM_VIDEO)) || Pn <= 0 || Pn > P ||
            Pn % mm || S <= 0 || !batch->enc_seg_h || !batch->enc_seg_w) {
            rad_step_fail(c, "hyv4flash: an encoder pass this program's tower cannot serve: no "
                             "tower, more patches than it was declared for, not whole merged "
                             "rows, or no segment grids");
            return false;
        }
        /* The layout, checked before anything is issued: every segment whole merged blocks. */
        int64_t merged = 0, rows_out = 0, max_h = 0;
        for (int64_t s = 0; s < S; ++s) {
            const int64_t h = batch->enc_seg_h[s], w = batch->enc_seg_w[s];
            if (h <= 0 || w <= 0 || h % merge || w % merge) {
                rad_step_fail(c, "hyv4flash: a tower segment is not whole merged blocks");
                return false;
            }
            merged += (h / merge) * (w / merge);
            rows_out += (h / merge) * (w / merge + 1);
            max_h = std::max(max_h, h / merge);
        }
        if (merged != Rn || rows_out > Rout) {
            rad_step_fail(c, "hyv4flash: the pass's segment grids do not add up to its patches");
            return false;
        }
        const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
        const RadOperand crd = praw2(batch->enc_coord, RAD_I32, 4, Pn);
        const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

        RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), RAD_W(b_patch), RAD_NONE, brows(b_x, Pn));
        RAD_ISSUE_N(c, op_pos, Pn, brows(b_x, Pn), RAD_W(w_pos), crd);
        t_embed.step(c, brows(b_x, Pn), Pn);
        for (size_t i = 0; i < L.size(); ++i) {
            const Layer& l = L[i];
            RAD_ISSUE_N(c, l.ln1, Pn, brows(b_x, Pn), RAD_W(l.n1w), RAD_W(l.n1b), brows(b_h, Pn));
            RAD_ISSUE_N(c, l.qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                        brows(b_qkv, Pn));
            RAD_ISSUE_N(c, l.attn, Pn, bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                        bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
            RAD_ISSUE_N(c, l.proj, Pn, brows(b_att, Pn), RAD_W(l.o_w), RAD_W(l.o_b),
                        brows(b_x, Pn), brows(b_x, Pn));
            RAD_ISSUE_N(c, l.ln2, Pn, brows(b_x, Pn), RAD_W(l.n2w), RAD_W(l.n2b), brows(b_h, Pn));
            RAD_ISSUE_N(c, l.up, Pn, brows(b_h, Pn), RAD_W(l.up_w), RAD_W(l.up_b), RAD_NONE,
                        brows(b_ff, Pn));
            RAD_ISSUE_N(c, l.down, Pn, brows(b_ff, Pn), RAD_W(l.down_w), RAD_W(l.down_b),
                        brows(b_x, Pn), brows(b_x, Pn));
            t_blk[i].step(c, brows(b_x, Pn), Pn);
        }
        /* The norm writes Pn rows of width; the unfold reads them as Pn / m kernel rows. */
        RAD_ISSUE_N(c, op_rms, Pn, brows(b_x, Pn), RAD_W(w_rms), brows(b_post, Pn / merge));
        RAD_ISSUE_N(c, op_unfold, Pn / merge, brows(b_post, Pn / merge), cu, RAD_NONE,
                    brows(b_unf, Rn));
        RAD_ISSUE_N(c, op_m0, Rn, brows(b_unf, Rn), RAD_W(w_m0), RAD_W(b_m0), RAD_NONE,
                    brows(b_m, Rn));
        /* The newline column: max_h rows of the newline, a lookup of row 0 of its table. */
        RAD_ISSUE_N(c, op_nl_ids, max_h, brows(b_zero, max_h));
        RAD_ISSUE_N(c, op_nl, max_h, brows(b_zero, max_h), RAD_W(w_nl), brows(b_nl, max_h));
        RAD_ISSUE_N(c, op_m2, Rn, brows(b_m, Rn), RAD_W(w_m2), RAD_W(b_m2), RAD_NONE,
                    brows(b_r, Rn));
        t_merged.step(c, brows(b_r, Rn), Rn);

        /* The layout: per segment one pitched copy of its merged rows and one of its newline
         * cells, on the pass's stream behind the GEMM. */
        char* out = (char*)rad_buf_ptr(c, b_out);
        const char* nl = (const char*)rad_buf_ptr(c, b_nl);
        const char* r = (const char*)rad_buf_ptr(c, b_r);
        const int64_t rowb = n_embd * 2;
        RadStream st = rad_stream(c);
        if (!out || !nl || !r) {
            rad_step_fail(c, "hyv4flash: the tower's layout buffers did not bind");
            return false;
        }
        int64_t m0 = 0, o0 = 0;
        for (int64_t s = 0; s < S; ++s) {
            const int64_t hs = batch->enc_seg_h[s] / merge, ws = batch->enc_seg_w[s] / merge;
            const int64_t pitch = (ws + 1) * rowb;
            if (rad_memcpy_2d_async(out + o0 * rowb, pitch, r + m0 * rowb, ws * rowb, ws * rowb,
                                    hs, st) < 0 ||
                rad_memcpy_2d_async(out + (o0 + ws) * rowb, pitch, nl, rowb, rowb, hs, st) < 0) {
                rad_step_fail(c, "hyv4flash: the tower's rows could not be laid out");
                return false;
            }
            m0 += hs * ws;
            o0 += hs * (ws + 1);
        }
        return true;
    }
};

}  // namespace hyv4flash
#endif
