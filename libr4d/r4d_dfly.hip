// r4d_dfly.hip -- dfly_sel: the page table and length a DFly2 draft row attends through.
//
//   For sequence s (rows [cu[s], cu[s+1])): a = pos[cu[s]], g0 = a > 0 ? (a-1)/ratio : 0,
//   k = src[s], n = nsel[k]. n > 0: nrec = n-1, out[r][c] = sel[k][c] for c < nrec (the source's
//   recalled pages; its own page is bt[g0]); n = 0: nrec = g0, out[r][c] = bt[s][c] (a densely
//   attended source). e = pos[r]/ratio - g0; out[r][nrec+j] = bt[s][g0+j] for j = 0..e; -1 to
//   `blocks`. len[r] = nrec*ratio + pos[r] - g0*ratio + 1; count[r] = nrec + e + 1 (optional).
//   A source of -1 or outside `sel`, a table past `blocks`, a recalled count past `blocks_in` or a
//   page past bt's pitch: the row is -1 and len (count) 0. Rows no sequence holds likewise.
//
// libref/ref_dfly.cpp is the oracle (byte for byte), kernels/dfly2.md §4 the design; written for
// this repository.
//
// One workgroup of 64 threads a (row, 64-column chunk): every thread recomputes the row's scalars
// (a, g0, k, n, nrec, e) from pos, cu, src and nsel, finds the row's sequence by a binary search of
// cu and writes its own column. No LDS, no scratch, no barrier. len and count are written by the
// thread of column 0. Columns of `out` at and past `blocks` are not touched.
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <cstdint>

namespace {

enum { kDfThreads = 64 };

struct DflySel {
    const int32_t* sel;
    const int32_t* nsel;
    const int32_t* src;
    const int32_t* bt;
    const int32_t* pos;
    const int32_t* cu;
    int32_t* out;
    int32_t* len;
    int32_t* cnt;          // may be null
    int64_t sp, bp, op;    // row pitches of sel, bt, out
    int32_t M, R, n_seq, pitch, blocks, blocks_in, ratio;
};

// The sequence a row belongs to: the last s with cu[s] <= m. -1 outside [cu[0], cu[n_seq]).
__device__ __forceinline__ int32_t df_seq_of(const int32_t* __restrict__ cu, int32_t n_seq,
                                             int32_t m) {
    if (m < cu[0] || m >= cu[n_seq]) return -1;
    int32_t lo = 0, hi = n_seq;  // cu[lo] <= m < cu[hi]
    while (hi - lo > 1) {
        const int32_t mid = (lo + hi) >> 1;
        if (cu[mid] <= m) lo = mid;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ void dfly_sel_body(DflySel p) {
    const int32_t r = (int32_t)blockIdx.x;
    const int32_t c = (int32_t)(blockIdx.y * kDfThreads + threadIdx.x);
    if (r >= p.M) return;
    const int32_t s = df_seq_of(p.cu, p.n_seq, r);
    bool row = false;
    int64_t nrec = 0, e = 0, g0 = 0, k = -1, n = 0, ps = 0;
    if (s >= 0) {
        const int32_t r0 = p.cu[s] < 0 ? 0 : p.cu[s];
        const int64_t anc = p.pos[r0];
        g0 = anc > 0 ? (anc - 1) / p.ratio : 0;
        k = p.src[s];
        bool ok = k >= 0 && k < p.R && anc >= 0;
        n = ok ? (int64_t)p.nsel[k] : 0;
        nrec = n > 0 ? n - 1 : g0;
        if (ok && n > 0 && nrec > p.blocks_in) ok = false;
        ps = p.pos[r];
        e = ps / p.ratio - g0;
        row = ok && ps >= anc && e >= 0;
        if (row && (nrec + e + 1 > p.blocks || g0 + e >= p.pitch || (n <= 0 && nrec > p.pitch)))
            row = false;
    }
    if (c < p.blocks) {
        int32_t v = -1;
        if (row) {
            if (c < nrec)
                v = n > 0 ? p.sel[k * p.sp + c] : p.bt[(int64_t)s * p.bp + c];
            else if (c <= nrec + e)
                v = p.bt[(int64_t)s * p.bp + g0 + (c - nrec)];
        }
        p.out[(int64_t)r * p.op + c] = v;
    }
    if (c == 0) {
        p.len[r] = row ? (int32_t)(nrec * p.ratio + (ps - g0 * p.ratio) + 1) : 0;
        if (p.cnt) p.cnt[r] = row ? (int32_t)(nrec + e + 1) : 0;
    }
}

// CH `r4dchain`: the chained form (r4d_chain.h); the body's early return stays inside it.
template <class CH>
__global__ __launch_bounds__(kDfThreads) void dfly_sel_kernel(DflySel p) {
    if (CH::on) r4d_chain_wait();
    dfly_sel_body(p);
    if (CH::on) r4d_chain_done();
}

bool df_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }

}  // namespace

extern "C" int r4d_rad_dfly_sel(const RadArgs* a, RadStream s) {
    enum { SEL = 0, NSEL, SRC, BT, POS, CU, OUT, LEN, CNT };
    const RadTensor *sel = r4d_opt(a, SEL), *nsel = r4d_opt(a, NSEL), *src = r4d_opt(a, SRC),
                    *bt = r4d_opt(a, BT), *pos = r4d_opt(a, POS), *cu = r4d_opt(a, CU),
                    *out = r4d_opt(a, OUT), *len = r4d_opt(a, LEN), *cnt = r4d_opt(a, CNT);
    if (!sel || !nsel || !src || !bt || !pos || !cu || !out || !len) return R4D_NO(RAD_E_INVAL);
    if (!df_i32(sel) || !df_i32(nsel) || !df_i32(src) || !df_i32(bt) || !df_i32(pos) ||
        !df_i32(cu) || !df_i32(out) || !df_i32(len) || (cnt && !df_i32(cnt)))
        return R4D_NO(RAD_E_DTYPE);
    const long long ratio = r4d_geti_or(a, "ratio", 0), blocks = r4d_geti_or(a, "blocks", 0),
                    blocks_in = r4d_geti_or(a, "blocks_in", 0);
    if (ratio <= 0 || blocks <= 0 || blocks_in <= 0 || ratio > INT32_MAX || blocks > INT32_MAX ||
        blocks_in > INT32_MAX)
        return R4D_NO(RAD_E_SHAPE);
    if (sel->rank != 2 || bt->rank != 2 || out->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = out->shape[0], R = sel->shape[0], pitch = bt->shape[1];
    const long long n_seq = r4d_numel(cu) - 1;
    if (M <= 0 || n_seq <= 0 || M > INT32_MAX || R > INT32_MAX || n_seq > INT32_MAX ||
        pitch > INT32_MAX)
        return R4D_NO(RAD_E_SHAPE);
    if (out->shape[1] < blocks || sel->shape[1] < blocks_in) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(pos) < M || r4d_numel(len) < M || r4d_numel(src) < n_seq ||
        bt->shape[0] < n_seq || r4d_numel(nsel) < R || (cnt && r4d_numel(cnt) < M))
        return R4D_NO(RAD_E_SHAPE);
    if (sel->stride[1] != 1 || bt->stride[1] != 1 || out->stride[1] != 1 || !r4d_contig(nsel) ||
        !r4d_contig(src) || !r4d_contig(pos) || !r4d_contig(cu) || !r4d_contig(len) ||
        (cnt && !r4d_contig(cnt)))
        return R4D_NO(RAD_E_STRIDE);
    DflySel p{};
    p.sel = (const int32_t*)sel->data;
    p.nsel = (const int32_t*)nsel->data;
    p.src = (const int32_t*)src->data;
    p.bt = (const int32_t*)bt->data;
    p.pos = (const int32_t*)pos->data;
    p.cu = (const int32_t*)cu->data;
    p.out = (int32_t*)out->data;
    p.len = (int32_t*)len->data;
    p.cnt = cnt ? (int32_t*)cnt->data : nullptr;
    p.sp = sel->stride[0];
    p.bp = bt->stride[0];
    p.op = out->stride[0];
    p.M = (int32_t)M;
    p.R = (int32_t)R;
    p.n_seq = (int32_t)n_seq;
    p.pitch = (int32_t)pitch;
    p.blocks = (int32_t)blocks;
    p.blocks_in = (int32_t)blocks_in;
    p.ratio = (int32_t)ratio;
    const uint32_t chunks = (uint32_t)((blocks + kDfThreads - 1) / kDfThreads);
    return r4d_geti_or(a, "chain", 0) != 0
               ? R4D_EMIT(s, (dfly_sel_kernel<r4dchain>), dim3((uint32_t)M, chunks),
                          dim3(kDfThreads), 0, p)
               : R4D_EMIT(s, (dfly_sel_kernel<R4dUnchained>), dim3((uint32_t)M, chunks),
                          dim3(kDfThreads), 0, p);
}
