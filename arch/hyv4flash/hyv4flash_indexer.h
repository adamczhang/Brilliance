/* hyv4flash_indexer.h -- the compressed indexer of a sparse layer (HYV4FlashCompressedIndexer,
 * PLAN §1.4, kernels/indexer.md).
 *
 * Every step, for every sequence: the token keys (index_key, its norm), parked in the tail
 * (qsa_tail_store) and, for each group of `ratio` the step closes, staged (qsa_work), pooled in
 * f32, normed, rotated at the group's first position and written into the indexer cache
 * (idx_pool_gated): HF pools every group again each call, and pooling once when a group closes gives
 * the same values.
 *
 * On a step whose context passes the dense limit (group_topk * ratio): the query side
 * (index_query, served as two linears over its rows like q_b, the rope part rotated at the query's
 * position, index_head), the scores (idx_score) and the selection (qsa_select with topk =
 * group_topk - 1, signed, own 1: the query's own group is never a candidate and always appended).
 * The table and lengths it writes are what mla_attn_sel_sink reads: the latent group's pages are
 * the indexer's groups (both groups of block 4). At or below the limit HF attends densely, which is
 * what the selection would give (kernels/indexer.md §1), so nothing of the query side runs.
 *
 * RADIANCE_HY4_INDEXER=fp8 (Shape::ix_fp8) swaps the key cache and the score for vLLM's IndexPool
 * numerics on GLM's forms: idx_pool_gated_fp8 keeps the pooled, normed, rotated key in f32 and
 * writes its Hadamard-rotated E4M3 codes and power-of-two scale into dsa_pool's packed page (u8,
 * 136 bytes a group); the rotated query goes through dsa_query (the same Hadamard and quantiser,
 * per head) and dsa_score (16 heads, the head weights as f32 gates). RADIANCE_HY4_INDEXER=int8
 * (Shape::ix_i8, with ix_fp8 set as "the packed wiring") is the same wiring with symmetric int8
 * codes at an f32 scale amax / 127 a key and a query head row: idx_pool_gated_i8, dsa_query_i8 and
 * dsa_score at dtype int8 (iu8 WMMA, int32 sums). Selection is qsa_select's in all modes. The exact
 * path stays the default: HF is the parity reference, and both quantised forms move near-ties
 * (notes/wp14.md, notes/idxint8.md). */
#ifndef HYV4FLASH_INDEXER_H
#define HYV4FLASH_INDEXER_H

#include "hyv4flash_attn.h"

namespace hyv4flash {

/* The planes every sparse layer's indexer shares. */
struct IndexWire {
    rad_buf tk = 0;       /* [T, D] token keys */
    rad_buf stage = 0;    /* [work * ratio, D] the closing groups' raw keys */
    rad_buf page = 0;     /* [work] i32 */
    rad_buf bpos = 0;     /* [work] i32 the groups' first positions; [work, 3] under M-RoPE */
    rad_buf nc = 0;       /* [seqs] i32 closed groups a sequence */
    rad_buf iqn = 0;      /* [T, Hi * (D - R)] */
    rad_buf iqr = 0;      /* [T, Hi * R] */
    rad_buf iq = 0;       /* [T, Hi, D] the rotated query */
    rad_buf iw = 0;       /* [T, Hi] head weights */
    rad_buf score = 0;    /* [T, blocks] f32 */
    rad_buf sel = 0;      /* [T, group_topk] i32 page table a row */
    rad_buf nsel = 0;     /* [T] i32 */
    rad_buf len = 0;      /* [T] i32 */
    /* RADIANCE_HY4_INDEXER=fp8 / int8 only */
    rad_buf iqf = 0;      /* [T * Hi, D] the rotated query, a row a head (dsa_query's form) */
    rad_buf qcodes = 0;   /* [T * Hi, D] int8 */
    rad_buf qscale = 0;   /* [T * Hi, 1] f32 */
    rad_buf iwf = 0;      /* [T, Hi] f32 head weights (dsa_score's gates) */
    /* taps only */
    rad_buf zero_pos = 0; /* [work] i32 zeros: the pooled key before its rotation */
    rad_buf ident = 0;    /* [seqs, blocks] i32 the identity table: selections as group indices */
    rad_buf sel_ids = 0;  /* [T, group_topk] i32 */
    int64_t work = 0, blocks = 0, ring = 0;
    int declare(RadBuilder* b, Names& nm, const Shape& s, const Geom& g) {
        const int64_t T = g.max_tok, D = s.ix_dim, R = s.sparse.qk_rope, Hi = s.ix_heads;
        work = (T + s.ix_pool - 1) / s.ix_pool + g.max_seqs;
        blocks = (g.max_ctx + s.ix_pool - 1) / s.ix_pool + 1;
        ring = s.ix_pool + g.max_spec;
        /* RADIANCE_HY4_TEST_RING=<delta> (a test, WP13.0): a ring smaller or larger by delta rows,
         * to show the rollback gate's probes can fail. Never set in service. */
        if (const char* e = std::getenv("RADIANCE_HY4_TEST_RING"); e && *e)
            ring = std::max<int64_t>(1, ring + std::strtoll(e, nullptr, 10));
        tk = decl_b(b, nm.f("ix.token_keys"), RAD_BF16, { T, D });
        stage = decl_b(b, nm.f("ix.stage"), RAD_BF16, { work * s.ix_pool, D });
        page = decl_b(b, nm.f("ix.page"), RAD_I32, { work });
        /* Under M-RoPE a group rotates at its first token's three components, side by side as
         * qsa_work writes them from the step's rotary planes and idx_pool_gated reads them. */
        bpos = s.mrope ? decl_b(b, nm.f("ix.bpos"), RAD_I32, { work, 3 })
                       : decl_b(b, nm.f("ix.bpos"), RAD_I32, { work });
        nc = decl_b(b, nm.f("ix.nc"), RAD_I32, { g.max_seqs });
        iqn = decl_b(b, nm.f("ix.query_nope"), RAD_BF16, { T, Hi * (D - R) });
        iqr = decl_b(b, nm.f("ix.query_rope"), RAD_BF16, { T, Hi * R });
        iq = decl_b(b, nm.f("ix.query"), RAD_BF16, { T, Hi, D });
        iw = decl_b(b, nm.f("ix.head_weights"), RAD_BF16, { T, Hi });
        score = decl_b(b, nm.f("ix.score"), RAD_F32, { T, blocks });
        sel = decl_b(b, nm.f("ix.selected"), RAD_I32, { T, s.ix_group_topk });
        nsel = decl_b(b, nm.f("ix.nsel"), RAD_I32, { T });
        len = decl_b(b, nm.f("ix.length"), RAD_I32, { T });
        if (!tk || !stage || !page || !bpos || !nc || !iqn || !iqr || !iq || !iw || !score ||
            !sel || !nsel || !len)
            return RAD_E_INVAL;
        if (s.ix_fp8) {
            iqf = decl_b(b, nm.f("ix.query_rows"), RAD_BF16, { T * Hi, D });
            qcodes = decl_b(b, nm.f("ix.query_codes"), s.ix_i8 ? RAD_I8 : RAD_F8E4M3, { T * Hi, D });
            qscale = decl_b(b, nm.f("ix.query_scales"), RAD_F32, { T * Hi, 1 });
            iwf = decl_b(b, nm.f("ix.head_gates"), RAD_F32, { T, Hi });
            if (!iqf || !qcodes || !qscale || !iwf) return RAD_E_INVAL;
        }
        if (taps_on()) {
            /* PERSIST: a tap's scratch is not in the declared op order's liveness. */
            zero_pos = decl_b(b, nm.f("ix.zero_pos"), RAD_I32, { work }, RAD_BUF_PERSIST);
            ident = decl_b(b, nm.f("ix.identity"), RAD_I32, { g.max_seqs, blocks }, RAD_BUF_PERSIST);
            sel_ids = decl_b(b, nm.f("ix.selected_ids"), RAD_I32, { T, s.ix_group_topk },
                             RAD_BUF_PERSIST);
            if (!zero_pos || !ident || !sel_ids) return RAD_E_INVAL;
        }
        return RAD_OK;
    }
};

struct Indexer {
    Shape s{};
    Geom g{};
    IndexWire w{};
    int layer = 0;
    rad_buf xa = 0, qa = 0;
    rad_kvgroup index = 0, tail = 0, attn = 0;
    Lin key{}, query_nope{}, query_rope{}, head{};
    rad_weight key_gain = 0, wv = 0, wg = 0, ape = 0, pool_gain = 0;
    rad_op op_key_norm = 0, op_work = 0, op_pool = 0, op_tail = 0;
    rad_op op_q_rope = 0, op_qn_copy = 0, op_qr_copy = 0, op_score = 0, op_select = 0;
    rad_op op_q_rows = 0, op_q_quant = 0, op_gate_f32 = 0;   /* RADIANCE_HY4_INDEXER=int8 */
    rad_op op_tap_zero = 0, op_tap_pool = 0, op_tap_ident = 0, op_tap_select = 0, op_tap_nsel0 = 0;
    Tap t_token_keys{}, t_group_keys{}, t_sel{}, t_nsel{}, t_score{};

    /* `index` is the indexer cache group, declared by the caller right before this (block 4,
     * claimed by idx_pool_gated); `tail` the raw-key ring, `attention` the sparse latent group. */
    int declare(RadBuilder* b, Names& nm, const Shape& shape, const Geom& geom, int l,
                const IndexWire& wire, rad_buf in, rad_buf q_resid, rad_kvgroup index_group,
                rad_kvgroup tail_group, rad_kvgroup attention, const char* ap,
                bool first_of_group) {
        s = shape;
        g = geom;
        w = wire;
        layer = l;
        xa = in;
        qa = q_resid;
        index = index_group;
        tail = tail_group;
        attn = attention;
        const int64_t T = g.max_tok, D = s.ix_dim, R = s.sparse.qk_rope, Hi = s.ix_heads,
                      r = s.ix_pool;
        const auto grp = grp_layer(l);
        const char* ip = nm.ckpt("%s.compressed_indexer", ap);
        auto act = [](rad_buf x) {
            ActFP8 v{};
            v.x = x;
            return v;
        };
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, tail));

        RAD_ARCH_TRY(key.declare(b, nm, g, nm.f("blk.%d.index_key", l), D, s.n_embd,
                                 RAD_SHARD_NONE, grp,
                                 { nm.ckpt("%s.token_key_projection.weight", ip) }, act(xa), w.tk));
        key_gain = decl_f32(b, nm.f("blk.%d.index_key_norm.weight", l),
                            nm.ckpt("%s.token_key_norm.weight", ip), { D }, grp);
        wv = decl_f32(b, nm.f("blk.%d.index_pool_value.weight", l),
                      nm.ckpt("%s.key_value_projection.weight", ip), { D, D }, grp);
        wg = decl_f32(b, nm.f("blk.%d.index_pool_gate.weight", l),
                      nm.ckpt("%s.key_gate_projection.weight", ip), { D, D }, grp);
        pool_gain = decl_f32(b, nm.f("blk.%d.index_pool_norm.weight", l),
                             nm.ckpt("%s.key_norm.weight", ip), { D }, grp);
        ape = decl_f32(b, nm.f("blk.%d.index_ape", l), nm.ckpt("%s.intra_group_ape", ip), { r, D },
                       grp);
        if (!key_gain || !wv || !wg || !pool_gain || !ape) return RAD_E_INVAL;
        op_key_norm = decl_rmsnorm(b, T, D, s.eps, key_gain, w.tk, w.tk);
        HY4_NEED(op_key_norm, "rmsnorm");

        op_work = mk(b, "qsa_work",
                     { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("ratio", r),
                       RAD_INT("ring", w.ring), RAD_INT("seqs", g.max_seqs), RAD_INT("work", w.work),
                       RAD_STR("dtype", "bf16") },
                     {}, { w.tk }, { w.stage, w.page, w.bpos, w.nc });
        HY4_NEED(op_work, "qsa_work");
        {
            /* Under M-RoPE a group turns at its first token's three components (qsa_work's bpos
             * [work, 3]); the FP8 form shares the pooling and its positions. */
            const char* pool_op = s.ix_i8    ? "idx_pool_gated_i8"
                                  : s.ix_fp8 ? "idx_pool_gated_fp8" : "idx_pool_gated";
            op_pool = s.mrope
                ? mk(b, pool_op,
                     { RAD_RANGE("M", 1, w.work), RAD_INT("n", D), RAD_INT("ratio", r),
                       RAD_INT("rotary_dim", R), RAD_F64("eps", s.eps),
                       RAD_F64("theta", s.theta_sparse), RAD_STR("mode", "imrope_gptj"),
                       RAD_STR("sections", s.mrope_sparse.c_str()), RAD_INT("block_size", r),
                       RAD_STR("dtype", "bf16") },
                     { wv, wg, ape, pool_gain }, { w.stage, w.bpos, w.page }, {})
                : mk(b, pool_op,
                     { RAD_RANGE("M", 1, w.work), RAD_INT("n", D), RAD_INT("ratio", r),
                       RAD_INT("rotary_dim", R), RAD_F64("eps", s.eps),
                       RAD_F64("theta", s.theta_sparse), RAD_STR("mode", "gptj"),
                       RAD_INT("block_size", r), RAD_STR("dtype", "bf16") },
                     { wv, wg, ape, pool_gain }, { w.stage, w.bpos, w.page }, {});
        }
        HY4_NEED(op_pool, s.ix_i8    ? "idx_pool_gated_i8 (RADIANCE_HY4_INDEXER=int8)"
                          : s.ix_fp8 ? "idx_pool_gated_fp8 (RADIANCE_HY4_INDEXER=fp8)"
                                     : "idx_pool_gated");
        if (first_of_group && rad_kv_block_size(b, index) != r) {
            fprintf(stderr, "radiance: hyv4flash: the indexer cache took a block of %lld tokens; "
                            "a page is one group of %lld\n",
                    (long long)rad_kv_block_size(b, index), (long long)r);
            return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, index));
        op_tail = mk(b, "qsa_tail_store",
                     { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("ratio", r),
                       RAD_INT("ring", w.ring), RAD_INT("seqs", g.max_seqs),
                       RAD_STR("dtype", "bf16") },
                     {}, { w.tk }, {});
        HY4_NEED(op_tail, "qsa_tail_store");

        /* The query side, like q_b: index_query's rows are head-major, a head's leading D - R
         * channels then its R rotated ones (HF's `_apply_partial_rotary_pos_emb`). */
        const char* qsrc = nm.ckpt("%s.query_projection.weight", ip);
        auto part = [&](const char* name, int64_t off, int64_t width) -> int {
            for (int64_t h0 = 0; h0 < Hi; h0 += 8) {
                RadNameMap m = map_base(name, RAD_MAP_ROWS);
                for (int64_t h = h0; h < std::min<int64_t>(h0 + 8, Hi); ++h) {
                    m.src[m.n_src] = qsrc;
                    m.src_index[m.n_src] = (int)(h * D + off);
                    m.src_rows[m.n_src] = width;
                    ++m.n_src;
                }
                RAD_ARCH_TRY(rad_decl_name_map(b, &m));
            }
            return RAD_OK;
        };
        RAD_ARCH_TRY(part(nm.f("blk.%d.index_query_nope.weight", l), 0, D - R));
        RAD_ARCH_TRY(part(nm.f("blk.%d.index_query_rope.weight", l), D - R, R));
        query_nope.mapped = query_rope.mapped = true;
        RAD_ARCH_TRY(query_nope.declare(b, nm, g, nm.f("blk.%d.index_query_nope", l), Hi * (D - R),
                                        s.q_lora, RAD_SHARD_NONE, grp, { qsrc }, act(qa), w.iqn));
        RAD_ARCH_TRY(query_rope.declare(b, nm, g, nm.f("blk.%d.index_query_rope", l), Hi * R,
                                        s.q_lora, RAD_SHARD_NONE, grp, { qsrc }, act(qa), w.iqr));
        /* K-49: Hi heads of rope a row, so over a prefill chunk this is the other rope where the
         * row form divides the `sincosf` by the head count. The drafter's own rope (dfly.h:634)
         * is the same op over the draft block's rows -- min(max_tok, seqs * block), dfly.h:358 --
         * and K-49 measured the trunk's and the indexer's, so it stays on `rope`. */
        op_q_rope = decl_rope_gptj(b, T, Hi, R, s.theta_sparse, w.iqr,
                                   s.mrope ? s.mrope_sparse.c_str() : nullptr, "rope_rows");
        op_qn_copy = decl_copy(b, T, Hi * (D - R), RAD_BF16, w.iqn, w.iq);
        op_qr_copy = decl_copy(b, T, Hi * R, RAD_BF16, w.iqr, w.iq);
        HY4_NEED(op_q_rope && op_qn_copy && op_qr_copy, "rope / cast");
        RAD_ARCH_TRY(head.declare(b, nm, g, nm.f("blk.%d.index_head", l), Hi, s.n_embd,
                                  RAD_SHARD_NONE, grp,
                                  { nm.ckpt("%s.head_weight_projection.weight", ip) }, act(xa),
                                  w.iw));
        if (s.ix_fp8) {
            /* The rotated query a row a head, its E4M3 codes and scales, the head weights as
             * f32 gates; dsa_score's (heads n)^-0.5 is HF's head weight scale. */
            op_q_rows = mk(b, "cast",
                           { RAD_RANGE("M", 1, T * Hi), RAD_INT("n", D), RAD_STR("from", "bf16"),
                             RAD_STR("to", "bf16") },
                           {}, { w.iq }, { w.iqf });
            op_q_quant = mk(b, s.ix_i8 ? "dsa_query_i8" : "dsa_query",
                            { RAD_RANGE("M", 1, T * Hi), RAD_INT("n", D), RAD_STR("dtype", "bf16") },
                            {}, { w.iqf }, { w.qcodes, w.qscale });
            op_gate_f32 = mk(b, "cast",
                             { RAD_RANGE("M", 1, T), RAD_INT("n", Hi), RAD_STR("from", "bf16"),
                               RAD_STR("to", "f32") },
                             {}, { w.iw }, { w.iwf });
            op_score = mk(b, "dsa_score",
                          { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("heads", Hi),
                            RAD_INT("blocks", w.blocks), RAD_INT("seqs", g.max_seqs),
                            RAD_STR("dtype", s.ix_i8 ? "int8" : "fp8_e4m3") },
                          {}, { w.qcodes, w.qscale, w.iwf, w.nc }, { w.score });
            HY4_NEED(op_q_rows && op_gate_f32, "cast");
            HY4_NEED(op_q_quant, "dsa_query / dsa_query_i8 (RADIANCE_HY4_INDEXER=int8)");
            HY4_NEED(op_score, "dsa_score at 16 heads (RADIANCE_HY4_INDEXER=int8)");
        } else {
            op_score = mk(b, "idx_score",
                          { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("heads", Hi),
                            RAD_INT("blocks", w.blocks), RAD_INT("seqs", g.max_seqs),
                            RAD_STR("dtype", "bf16") },
                          {}, { w.iq, w.iw, w.nc }, { w.score });
            HY4_NEED(op_score, "idx_score");
        }
        op_select = mk(b, "qsa_select",
                       { RAD_RANGE("M", 1, T), RAD_INT("blocks", w.blocks),
                         RAD_INT("topk", s.ix_group_topk - 1), RAD_INT("ratio", r),
                         RAD_INT("seqs", g.max_seqs), RAD_INT("signed", 1), RAD_INT("own", 1) },
                       {}, { w.score, w.nc }, { w.sel, w.nsel, w.len });
        HY4_NEED(op_select, "qsa_select (own)");

        /* Taps: the token keys; each closing group's pooled key before its rotation (the pool
         * again at position 0, which turns nothing); the selection as group indices (the select
         * again over an identity table) and its count. */
        RAD_ARCH_TRY(t_token_keys.declare(b, nm, T, "index_token_keys", l, RAD_BF16, D, w.tk));
        if (taps_on()) {
            const char* gk = nm.f("dbg.index_group_keys.%d", l);
            t_group_keys.buf = decl_b(b, gk, RAD_BF16, { std::max(T, w.work), D }, RAD_BUF_PERSIST);
            t_group_keys.width = D;
            op_tap_zero = mk(b, "iota",
                             { RAD_RANGE("M", 1, w.work), RAD_INT("start", 0), RAD_INT("step", 0) },
                             {}, {}, { w.zero_pos });
            op_tap_pool = mk(b, "idx_pool_gated",
                             { RAD_RANGE("M", 1, w.work), RAD_INT("n", D), RAD_INT("ratio", r),
                               RAD_INT("rotary_dim", R), RAD_F64("eps", s.eps),
                               RAD_F64("theta", s.theta_sparse), RAD_STR("mode", "gptj"),
                               RAD_STR("dtype", "bf16") },
                             { wv, wg, ape, pool_gain }, { w.stage, w.zero_pos },
                             { t_group_keys.buf });
            op_tap_ident = mk(b, "iota",
                              { RAD_RANGE("M", 1, g.max_seqs * w.blocks), RAD_INT("start", 0),
                                RAD_INT("step", 1), RAD_INT("period", w.blocks) },
                              {}, {}, { w.ident });
            op_tap_select = mk(b, "qsa_select",
                               { RAD_RANGE("M", 1, T), RAD_INT("blocks", w.blocks),
                                 RAD_INT("topk", s.ix_group_topk - 1), RAD_INT("ratio", r),
                                 RAD_INT("seqs", g.max_seqs), RAD_INT("signed", 1),
                                 RAD_INT("own", 1) },
                               {}, { w.score, w.nc, w.ident }, { w.sel_ids, w.len });
            HY4_NEED(t_group_keys.buf && op_tap_zero && op_tap_pool && op_tap_ident &&
                         op_tap_select,
                     "the indexer taps (iota, idx_pool_gated, qsa_select)");
            RAD_ARCH_TRY(t_sel.declare(b, nm, T, "index_sel", l, RAD_I32, s.ix_group_topk,
                                       w.sel_ids));
            RAD_ARCH_TRY(t_nsel.declare(b, nm, T, "index_nsel", l, RAD_I32, 1, w.nsel));
            /* The scores the selection read, a row's groups by index (WP4: the selection is
             * checked against its own scores, the scores against HF's). */
            RAD_ARCH_TRY(t_score.declare(b, nm, T, "index_score", l, RAD_F32, w.blocks, w.score));
            op_tap_nsel0 = mk(b, "fill",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", 1), RAD_STR("dtype", "i32") },
                              {}, {}, { t_nsel.buf });
            HY4_NEED(op_tap_nsel0, "fill");
        }
        return RAD_OK;
    }

    /* The key side, every step; the query side and the selection on a selecting step. Issued after
     * the attention's front (it reads q_resid, `qa`). */
    bool step(RadCtx* c, const RadBatch* batch, const StepPlan& p, const RadBatch* queries, QueryRows rows) const {
        const auto* ib = kv_batch(batch, index);
        const auto* tb = kv_batch(batch, tail);
        const auto* ab = kv_batch(batch, attn);
        if (!ib || !tb || !ab || !ib->block_table || !ab->block_table || !tb->state_index) {
            rad_step_fail(c, "hyv4flash: the indexer needs its cache's and the latent cache's "
                             "block tables and the tail's state slots");
            return false;
        }
        const int64_t T = batch->n_tok, S = p.S, D = s.ix_dim, R = s.sparse.qk_rope;
        const int64_t work = (T + s.ix_pool - 1) / s.ix_pool + S;
        const int64_t pitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
        auto act = [](rad_buf x) {
            ActFP8 v{};
            v.x = x;
            return v;
        };
        const RadOperand pos = praw(batch->positions, RAD_I32, T);
        const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
        /* M-RoPE: the step's rotary planes into qsa_work, which writes each group's three
         * components; a step without them (no encoder declared) cannot occur under s.mrope. */
        if (s.mrope && !batch->rope_pos) {
            rad_step_fail(c, "hyv4flash: the program rotates at M-RoPE positions and the step "
                             "carries no rope_pos");
            return false;
        }
        const RadOperand rope = s.mrope ? praw2(batch->rope_pos, RAD_I32, 3, T) : RAD_NONE;
        key.step(c, act(xa), w.tk, T);
        RAD_ISSUE_N(c, op_key_norm, T, brows(w.tk, T), RAD_W(key_gain), brows(w.tk, T));
        t_token_keys.step(c, brows(w.tk, T), T);
        RAD_ISSUE_N(c, op_work, T, brows(w.tk, T), pos, cu, kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch),
                    praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), rope,
                    brows(w.stage, work * s.ix_pool), brows(w.page, work), brows(w.bpos, work),
                    brows(w.nc, S));
        if (s.ix_fp8)
            RAD_ISSUE_N(c, op_pool, work, brows(w.stage, work * s.ix_pool), RAD_W(wv), RAD_W(wg),
                        RAD_W(ape), RAD_W(pool_gain), brows(w.bpos, work), brows(w.page, work),
                        kv_cache(index, layer), RAD_NONE, RAD_NONE);
        else
            RAD_ISSUE_N(c, op_pool, work, brows(w.stage, work * s.ix_pool), RAD_W(wv), RAD_W(wg),
                        RAD_W(ape), RAD_W(pool_gain), brows(w.bpos, work), brows(w.page, work),
                        kv_cache(index, layer), RAD_NONE);
        if (op_tap_pool) {
            RAD_ISSUE_N(c, op_tap_zero, work, brows(w.zero_pos, work));
            RAD_ISSUE_N(c, op_tap_pool, work, brows(w.stage, work * s.ix_pool), RAD_W(wv),
                        RAD_W(wg), RAD_W(ape), RAD_W(pool_gain), brows(w.zero_pos, work), RAD_NONE,
                        RAD_NONE, brows(t_group_keys.buf, work));
        }
        /* After the gather: the tail's rows are what the next step's groups read. */
        RAD_ISSUE_N(c, op_tail, T, brows(w.tk, T), pos, cu, kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch));
        if (!p.select || !rows.n) {
            if (op_tap_nsel0) RAD_ISSUE_N(c, op_tap_nsel0, T, brows(t_nsel.buf, T));
            return true;
        }
        const int64_t M = rows.n;
        const RadOperand qcu = praw(queries->cu_seqlens, RAD_I32, S + 1);
        const RadOperand qpos = praw(queries->positions, RAD_I32, M);
        query_nope.step(c, act(qa), w.iqn, M);
        query_rope.step(c, act(qa), w.iqr, M);
        RAD_ISSUE_N(c, op_q_rope, M, brows(w.iqr, M), rope_pos_of(s, queries, M));
        RadOperand qn = brows(w.iq, M), qr = qn;
        qn.cols = D - R;
        qr.offset = D - R;
        qr.cols = R;
        RAD_ISSUE_N(c, op_qn_copy, M, brows(w.iqn, M), qn);
        RAD_ISSUE_N(c, op_qr_copy, M, brows(w.iqr, M), qr);
        rows.spans([&](int64_t src, int64_t dst, int64_t count) {
            head.step(c, act(xa), w.iw, T, src, count, dst);
        });
        const RadOperand score = bcol_at(w.score, 0, w.blocks, 0, p.live, M);
        if (s.ix_fp8) {
            const int64_t Hi = s.ix_heads;
            RAD_ISSUE_N(c, op_q_rows, M * Hi, brows(w.iq, M), brows(w.iqf, M * Hi));
            RAD_ISSUE_N(c, op_q_quant, M * Hi, brows(w.iqf, M * Hi), brows(w.qcodes, M * Hi),
                        brows(w.qscale, M * Hi));
            RAD_ISSUE_N(c, op_gate_f32, M, brows(w.iw, M), brows(w.iwf, M));
            /* The packed cache in place of `k` and its scales (dsa_score's packed form). */
            RAD_ISSUE_N(c, op_score, M, brows(w.qcodes, M * Hi), brows(w.qscale, M * Hi),
                        brows(w.iwf, M), kv_cache(index, layer), RAD_NONE,
                        praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), qcu,
                        brows(w.nc, S), score, RAD_NONE);
        } else {
            RAD_ISSUE_N(c, op_score, M, brows(w.iq, M), brows(w.iw, M), kv_cache(index, layer),
                        praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), qcu,
                        brows(w.nc, S), score);
        }
        RAD_ISSUE_N(c, op_select, M, score,
                    praw2(ab->block_table, RAD_I32, S, ab->block_table_pitch), brows(w.nc, S), qpos,
                    qcu, brows(w.sel, M), brows(w.nsel, M), brows(w.len, M));
        if (op_tap_select) {
            /* The identity table is refilled each step; the select over it writes `len` again
             * with the same values (the lengths do not depend on the table). */
            RAD_ISSUE_N(c, op_tap_ident, S * w.blocks, bcol_at(w.ident, 0, w.blocks, 0, w.blocks, S));
            RAD_ISSUE_N(c, op_tap_select, M, score, bcol_at(w.ident, 0, w.blocks, 0, w.blocks, S),
                        brows(w.nc, S), qpos, qcu, brows(w.sel_ids, M), brows(w.nsel, M),
                        brows(w.len, M));
            t_sel.step(c, brows(w.sel_ids, M), M);
            t_nsel.step(c, brows(w.nsel, M), M);
            t_score.step(c, brows(w.score, M), M);
        }
        return true;
    }
};

}  // namespace hyv4flash
#endif
