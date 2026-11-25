/* ref_dfly.cpp -- dfly_sel: the page table and length a DFly2 draft row attends through
 * (docs/OPS.md "dfly_sel", kernels/dfly2.md §1.4 and §4).
 *
 * A block drafter's query pass runs `rows` positions a sequence, a..a+rows-1, past the committed end
 * a. Its attention layer i reads the latent cache of a trunk sparse layer and the keys that layer's
 * selection recalled at the trunk row of position a-1 (the source row, RadBatch::anchor_rows),
 * plus the own group of a-1 and the block's own rows, causally. With pages of `ratio` tokens
 * (one indexer group a page) that key set is
 *
 *     recalled closed pages (all before g0 = (a-1)/ratio)  U  tokens [ratio*g0, p]
 *
 * for the row at position p, and the second part is contiguous: the own page of a-1, then
 * p/ratio - g0 further pages. That is a table mla_attn_sel_sink reads (tokens [0, len) of a per-row
 * table): [recalled.., bt[g0], bt[g0+1], ..], len = ratio*nrec + (p - ratio*g0) + 1.
 *
 * The recalled pages come from the source row's snapshot of qsa_select's output (`sel`, page ids,
 * `nsel` entries, the own page last, which is not copied: it is bt[g0] and read from there), or,
 * for a source row that attended densely (`nsel` 0), from the block table: every closed page
 * before g0. Integer only; libavx and libr4d must give the same bytes. */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>

using namespace ref;

namespace {
int64_t li(const RadTensor* t, int64_t off) { return ld_int(t->dtype, t->data, off); }
void    si(const RadTensor* t, int64_t off, int64_t v) { st_int(t->dtype, t->data, off, v); }
}  // namespace

int ref_dfly_sel(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 8 || !have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sel = &a->t[0], *nsel = &a->t[1], *src = &a->t[2], *bt = &a->t[3];
    const RadTensor *pos = &a->t[4], *cu = &a->t[5], *out = &a->t[6], *len = &a->t[7];
    /* Optional: each row's entry count (the table's live prefix), kv_resolve's `nsel`. */
    const RadTensor* cnt = a->n_t > 8 && a->t[8].data ? &a->t[8] : nullptr;
    const int64_t ratio = p_int(a, "ratio", 0), blocks = p_int(a, "blocks", 0);
    const int64_t blocks_in = p_int(a, "blocks_in", 0);
    if (ratio <= 0 || blocks <= 0 || blocks_in <= 0) return RAD_E_SHAPE;
    if (sel->rank != 2 || bt->rank != 2 || out->rank != 2) return RAD_E_SHAPE;
    const int64_t M = out->shape[0], R = sel->shape[0], pitch = bt->shape[1];
    const int64_t n_seq = numel(cu) - 1;
    if (M <= 0 || n_seq <= 0) return RAD_E_SHAPE;
    if (out->shape[1] < blocks || sel->shape[1] < blocks_in) return RAD_E_SHAPE;
    if (numel(pos) < M || numel(len) < M || numel(src) < n_seq || bt->shape[0] < n_seq ||
        numel(nsel) < R || (cnt && numel(cnt) < M))
        return RAD_E_SHAPE;
    const int64_t o_rs = out->stride[0], o_cs = out->stride[1];
    const int64_t s_rs = sel->stride[0], s_cs = sel->stride[1];
    const int64_t b_rs = bt->stride[0], b_cs = bt->stride[1];
    bool overflow = false;
    /* Rows no sequence holds stay empty. */
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t c = 0; c < blocks; ++c) si(out, r * o_rs + c * o_cs, -1);
        si(len, offlin(len, r), 0);
        if (cnt) si(cnt, offlin(cnt, r), 0);
    }
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t r0 = std::max<int64_t>(0, cu_at(cu, s)), r1 = std::min(M, cu_at(cu, s + 1));
        if (r0 >= r1) continue;
        const int64_t anc = li(pos, offlin(pos, r0));
        const int64_t g0 = anc > 0 ? (anc - 1) / ratio : 0;
        const int64_t k = li(src, offlin(src, s));
        int64_t n = 0;
        bool ok = k >= 0 && k < R && anc >= 0;
        if (ok) n = li(nsel, offlin(nsel, k));
        const int64_t nrec = n > 0 ? n - 1 : g0;
        if (ok && n > 0 && nrec > blocks_in) {
            ok = false;
            overflow = true;
        }
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t p = li(pos, offlin(pos, r));
            const int64_t e = p / ratio - g0;
            bool row = ok && p >= anc && e >= 0;
            if (row && (nrec + e + 1 > blocks || g0 + e >= pitch || (n <= 0 && nrec > pitch))) {
                row = false;
                overflow = true;
            }
            if (!row) continue;   /* already empty */
            int64_t c = 0;
            if (n > 0)
                for (; c < nrec; ++c) si(out, r * o_rs + c * o_cs, li(sel, k * s_rs + c * s_cs));
            else
                for (; c < nrec; ++c) si(out, r * o_rs + c * o_cs, li(bt, s * b_rs + c * b_cs));
            for (int64_t j = 0; j <= e; ++j, ++c)
                si(out, r * o_rs + c * o_cs, li(bt, s * b_rs + (g0 + j) * b_cs));
            for (; c < blocks; ++c) si(out, r * o_rs + c * o_cs, -1);
            si(len, offlin(len, r), nrec * ratio + (p - g0 * ratio) + 1);
            if (cnt) si(cnt, offlin(cnt, r), nrec + e + 1);
        }
    }
    /* A table the caller sized too narrow fails its rows (len 0, the attention then gives 0 and the
     * proposal is verified anyway); a debug build also says so. */
#ifndef NDEBUG
    if (overflow) return RAD_E_SHAPE;
#else
    (void)overflow;
#endif
    return RAD_OK;
}
