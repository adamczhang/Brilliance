/* avx_ggml.cpp -- gemm_nt_ggml, moe_gemm_ggml, logits_gemm_ggml and embed_lookup_ggml over
 * llama.cpp's native blocks (DeepSeek-V4.1's linears, experts, head and embedding).
 *
 * The weight arrives as the layout hooks stored it (avx_ggml_hooks.cpp): u8 rows of
 * K / block * block_bytes bytes, rows back to back, at the declared [N, K] shape or as [N, row
 * bytes]; a float [N, K] (the logical form the checker passes) is read as it is. Each weight row
 * is dequantised once a call with llama.cpp's own dequantize_row (the baseline libquant code,
 * libref's too), then dotted with the activation rows at this level's width. libref accumulates in
 * index order; the dots here differ only in summation order and are checked to a tolerance.
 *
 * The activation rows are widened to f32 once a call, before the parallel region; the region
 * runs over weight rows (moe: over (expert, weight row) items), each output element written by
 * one thread in a fixed order, so a result does not depend on the team size. */
#include "avx_vec.h"
#include "avx_common.h"
#include "avx_ggml.h"

#include <cstring>
#include <vector>

using namespace avx;

namespace {

float dotv(const float* x, const float* y, int64_t n) {
    vf acc = vf_zero();
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) acc = vf_fma(vf_loadu(x + i), vf_loadu(y + i), acc);
    float s = vf_hsum(acc);
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}

struct Kind {
    int64_t block = 0, block_bytes = 0;
    AvxGgmlDequant dq = nullptr;
};

bool kind_of(const RadArgs* a, Kind* k) {
    return avx_ggml_type(p_str(a, "type", nullptr), &k->block, &k->block_bytes, &k->dq) == 1 &&
           k->block > 0 && k->dq;
}

/* Row n of a weight: native blocks dequantised, or the float logical form widened. */
void weight_row(const RadTensor& w, bool native, const Kind& k, int64_t rb, int64_t n, int64_t K,
                float* out) {
    if (native) {
        k.dq((const uint8_t*)w.data + n * rb, out, K);
        return;
    }
    const int64_t b = rowoff(&w, n, K), s = laststride(&w);
    for (int64_t i = 0; i < K; ++i) out[i] = ldt(&w, b + i * s);
}

/* x's first M rows of K, widened. */
std::vector<float> widen_rows(const RadTensor& x, int64_t M, int64_t K) {
    std::vector<float> v((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m) {
        const int64_t b = rowoff(&x, m, K), s = laststride(&x);
        for (int64_t i = 0; i < K; ++i) v[(size_t)(m * K + i)] = ldt(&x, b + i * s);
    }
    return v;
}

int gemm_ggml_impl(const RadArgs* a, const char* nN, const char* nK) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    Kind k;
    const bool known = kind_of(a, &k);
    const int64_t M = band_rows(x.rank >= 1 ? x.shape[0] : 0, p_int(a, "M", 0));
    const int64_t N = p_int(a, nN, 0), K = p_int(a, nK, 0);
    if (!known || M <= 0 || N <= 0 || K <= 0 || K % k.block || x.rank != 2 || x.shape[1] != K ||
        y.rank != 2 || y.shape[1] != N || y.shape[0] < M || w.rank != 2 || w.shape[0] != N)
        return RAD_E_SHAPE;
    const int64_t rb = K / k.block * k.block_bytes;
    const bool native = w.dtype == RAD_U8;
    if (native ? (w.shape[1] != rb && w.shape[1] != K) : w.shape[1] != K) return RAD_E_SHAPE;
    const std::vector<float> xv = widen_rows(x, M, K);
    const int64_t ys = laststride(&y);
    AVX_PARALLEL_FOR_IF(N * K >= AVX_PAR_MIN)
    for (int64_t n = 0; n < N; ++n) {
        float* wr = scratch_f32((size_t)K);
        weight_row(w, native, k, rb, n, K, wr);
        for (int64_t m = 0; m < M; ++m)
            stt(&y, rowoff(&y, m, N) + n * ys, dotv(xv.data() + m * K, wr, K));
    }
    return RAD_OK;
}

/* Expert e of a RAD_OPD_WTAB operand (a pointer table marked by stride[0] == 0) or of a stacked
 * rank-3 tensor, as avx_moe.cpp and libref read it. */
RadTensor expert_entry(const RadTensor* w, int64_t e) {
    RadTensor t{};
    if (!w || !w->data || w->rank < 2 || e < 0 || e >= w->shape[0]) return t;
    t.dtype = w->dtype;
    t.rank = w->rank - 1;
    for (uint32_t d = 0; d + 1 < w->rank; ++d) {
        t.shape[d] = w->shape[d + 1];
        t.stride[d] = w->stride[d + 1];
    }
    if (w->stride[0] == 0) {
        t.data = ((void* const*)w->data)[e];
    } else {
        const int64_t bits = rad_dtype_bits(w->dtype);
        const int64_t nbits = bits * w->stride[0] * e;
        if (bits <= 0 || (nbits & 7)) return t;
        t.data = (void*)((const char*)w->data + (nbits >> 3));
    }
    return t;
}

}  // namespace

AVX_KERNEL(gemm_nt_ggml) { return gemm_ggml_impl(a, "N", "K"); }

AVX_KERNEL(logits_gemm_ggml) { return gemm_ggml_impl(a, "n_vocab", "n_embd"); }

AVX_KERNEL(embed_lookup_ggml) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &w = a->t[1], &x = a->t[2];
    Kind k;
    const bool known = kind_of(a, &k);
    const int64_t E = p_int(a, "n_embd", 0), V = p_int(a, "n_vocab", 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (!known || E <= 0 || V <= 0 || voff < 0 || E % k.block || w.rank != 2 || w.shape[0] < V)
        return RAD_E_SHAPE;
    const int64_t rb = E / k.block * k.block_bytes;
    const bool native = w.dtype == RAD_U8;
    if (native ? (w.shape[1] != rb && w.shape[1] != E) : w.shape[1] != E) return RAD_E_SHAPE;
    const int64_t M = numel(&x) / E;
    if (M <= 0 || numel(&tok) < M) return RAD_E_SHAPE;
    if (voff == 0 && a->world_size <= 1)
        for (int64_t m = 0; m < M; ++m)
            if (ldi(&tok, offlin(&tok, m)) >= V) return RAD_E_INVAL;
    AVX_PARALLEL_FOR_IF(M * E >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ldi(&tok, offlin(&tok, m)) - voff;
        const int64_t xb = rowoff(&x, m, E), xs = laststride(&x);
        if (id < 0 || id >= V) {
            for (int64_t i = 0; i < E; ++i) stt(&x, xb + i * xs, 0.0f);
            continue;
        }
        float* wr = scratch_f32((size_t)E);
        weight_row(w, native, k, rb, id, E, wr);
        for (int64_t i = 0; i < E; ++i) stt(&x, xb + i * xs, wr[i] * mult);
    }
    return RAD_OK;
}

AVX_KERNEL(moe_gemm_ggml) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor &A = a->t[0], &W = a->t[1], &sorted = a->t[2], &off = a->t[3], &Y = a->t[4];
    Kind k;
    const bool known = kind_of(a, &k);
    const int64_t N = p_int(a, "N", 0), K = p_int(a, "K", 0);
    const int64_t ne = p_int(a, "n_expert", 0), top_k = p_int(a, "top_k", 0);
    const char* ord = p_str(a, "a_order", "token");
    const bool sorted_a = ord && !std::strcmp(ord, "sorted");
    if (ord && !sorted_a && std::strcmp(ord, "token")) return RAD_E_UNSUPPORTED;
    if (!known || N <= 0 || K <= 0 || K % k.block || ne <= 0 || top_k <= 0 || A.rank != 2 ||
        A.shape[1] != K || Y.rank != 2 || Y.shape[1] != N || W.rank != 3 || W.shape[0] < ne ||
        W.shape[1] != N)
        return RAD_E_SHAPE;
    const int64_t T = Y.shape[0], AR = A.shape[0];
    if (T <= 0 || numel(&sorted) < T || numel(&off) < ne + 1) return RAD_E_SHAPE;
    const int64_t rb = K / k.block * k.block_bytes;
    const bool native = W.dtype == RAD_U8;
    if (native ? (W.shape[2] != rb && W.shape[2] != K) : W.shape[2] != K) return RAD_E_SHAPE;
    std::vector<RadTensor> we((size_t)ne);
    for (int64_t e = 0; e < ne; ++e) {
        we[(size_t)e] = expert_entry(&W, e);
        if (native && W.stride[0] != 0) we[(size_t)e].data = (uint8_t*)W.data + e * N * rb;
        if (!we[(size_t)e].data) return RAD_E_STRIDE;
    }
    /* Each row's expert and activation row (-1: a row of zeros), then the live rows by expert. */
    std::vector<int64_t> ex((size_t)T, -1), tk((size_t)T, 0);
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ldi(&sorted, offlin(&sorted, i));
        if (flat < 0) continue;
        int64_t lo = 0, hi = ne;
        while (lo + 1 < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (ldi(&off, offlin(&off, mid)) <= i) lo = mid; else hi = mid;
        }
        const int64_t t = sorted_a ? i : flat / top_k;
        if (t < AR) { ex[(size_t)i] = lo; tk[(size_t)i] = t; }
    }
    std::vector<std::vector<int64_t>> rows((size_t)ne);
    for (int64_t i = 0; i < T; ++i) {
        if (ex[(size_t)i] >= 0) rows[(size_t)ex[(size_t)i]].push_back(i);
        else for (int64_t n = 0; n < N; ++n) stt(&Y, rowoff(&Y, i, N) + n * laststride(&Y), 0.0f);
    }
    /* The activation rows a live slot reads, widened once: slot i reads row tk[i]. */
    std::vector<int64_t> slot_of_row((size_t)AR, -1);
    std::vector<int64_t> live;
    for (int64_t i = 0; i < T; ++i)
        if (ex[(size_t)i] >= 0 && slot_of_row[(size_t)tk[(size_t)i]] < 0) {
            slot_of_row[(size_t)tk[(size_t)i]] = (int64_t)live.size();
            live.push_back(tk[(size_t)i]);
        }
    std::vector<float> av(live.size() * (size_t)K);
    for (size_t j = 0; j < live.size(); ++j) {
        const int64_t b = rowoff(&A, live[j], K), s = laststride(&A);
        for (int64_t c = 0; c < K; ++c) av[j * (size_t)K + (size_t)c] = ldt(&A, b + c * s);
    }
    std::vector<std::pair<int64_t, int64_t>> items;
    for (int64_t e = 0; e < ne; ++e)
        if (!rows[(size_t)e].empty())
            for (int64_t n = 0; n < N; ++n) items.push_back({ e, n });
    const int64_t n_items = (int64_t)items.size(), ys = laststride(&Y);
    AVX_PARALLEL_FOR_IF(n_items * K >= AVX_PAR_MIN)
    for (int64_t it = 0; it < n_items; ++it) {
        const int64_t e = items[(size_t)it].first, n = items[(size_t)it].second;
        float* wr = scratch_f32((size_t)K);
        weight_row(we[(size_t)e], native, k, rb, n, K, wr);
        for (int64_t i : rows[(size_t)e]) {
            const float* xr = av.data() + (size_t)slot_of_row[(size_t)tk[(size_t)i]] * (size_t)K;
            stt(&Y, rowoff(&Y, i, N) + n * ys, dotv(xr, wr, K));
        }
    }
    return RAD_OK;
}
