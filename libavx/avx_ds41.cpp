/* avx_ds41.cpp -- DeepSeek-V4.1's text ops on the host (arch/deepseek41): the rotary over a
 * window of each head, the block-scaled cache round trip and entries, the compressed cache's
 * softmax pooling, the sparse latent attention with sinks and its indexer, and Engram's lookups.
 *
 * Each op is libref's (libref/ref_sparse_kv.cpp, ref_attn.cpp: ref_rope_rows, ref_gemm.cpp:
 * ref_embed_lookup_planes) with the same validation, refusals and rounding points; the
 * attention's and the indexer's dot products run at this level's width over rows widened once,
 * and differ from libref's only in summation order. The selections (index_candidates,
 * index_topk), the id maps, the entry packing and the quantise-dequantise round trip are exact.
 *
 * Every operand is addressed through avx_common.h's copies of libref's helpers, so strided views
 * read as libref reads them. */
#include "avx_vec.h"
#include "avx_common.h"
#include "rad_kv_fp8.h"

#include <algorithm>
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

/* y[0, n) += p x[0, n) */
void axpy(float* y, float p, const float* x, int64_t n) {
    const vf pv = vf_set1(p);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(y + i, vf_fma(pv, vf_loadu(x + i), vf_loadu(y + i)));
    for (; i < n; ++i) y[i] += p * x[i];
}

int64_t ld_at(const RadTensor& t, int64_t lin) { return ldi(&t, offlin(&t, lin)); }

/* The s with cu[s] <= row < cu[s+1] (libref: seq_of_row). */
int64_t seq_of_row(const RadTensor& cu, int64_t n_seq, int64_t row) {
    int64_t lo = 0, hi = n_seq;
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi) / 2;
        if (ld_at(cu, mid) <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

int64_t keep_words(int64_t width, int64_t block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

/* The k largest of v by value, ties to the lower index, as indices (fewer when v is shorter). */
std::vector<int64_t> top_indices(const std::vector<float>& v, int64_t k) {
    std::vector<int64_t> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int64_t)i;
    k = std::min<int64_t>(k, (int64_t)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t x, int64_t y) {
        return v[(size_t)x] > v[(size_t)y] || (v[(size_t)x] == v[(size_t)y] && x < y);
    });
    idx.resize((size_t)k);
    return idx;
}

/* ---- quant_roundtrip's arithmetic, libref's exactly */
int ceil_log2_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
}
float pow2_bits(int e) {
    const uint32_t u = (uint32_t)(e + 127) << 23;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
enum class QFmt { Fp8E4m3, Fp4E2m1 };
enum class QScale { Ue8m0, E4m3, F32 };
float e4m3(float v) { return rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3(v)); }
float block_scale(float amax, QFmt f, QScale s) {
    if (f == QFmt::Fp8E4m3) {
        amax = std::max(amax, 1e-4f);
        const float inv = 1.0f / 448.0f;
        return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
    }
    if (s == QScale::E4m3) {
        amax = std::max(amax, 6.0f * 0x1p-9f);
        return e4m3(amax / 6.0f);
    }
    amax = std::max(amax, 6.0f * 0x1p-126f);
    const float inv = 1.0f / 6.0f;
    return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
}
float e2m1_value(unsigned nib) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const float v = kMag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}
float roundtrip(float v, float scale, QFmt f) {
    if (f == QFmt::Fp8E4m3) return e4m3(std::clamp(v / scale, -448.0f, 448.0f)) * scale;
    return e2m1_value(e2m1_code(std::clamp(v / scale, -6.0f, 6.0f))) * scale;
}

}  // namespace

/* ------------------------------------------------------------------ ds41_rope_rows */
AVX_KERNEL(ds41_rope_rows) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &cs = a->t[1], &pos = a->t[2];
    const int64_t nh = p_int(a, "n_head", 0), hd = p_int(a, "head_dim", 0);
    const int64_t rot = p_int(a, "rot", 0), off = p_int(a, "off", -1);
    const int64_t inv = p_int(a, "inverse", -1);
    const char* mode = p_str(a, "mode", "");
    const bool gptj = !std::strcmp(mode, "gptj");
    if (!gptj && std::strcmp(mode, "neox")) return RAD_E_UNSUPPORTED;
    if (nh < 1 || hd < 2 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd ||
        (inv != 0 && inv != 1))
        return RAD_E_SHAPE;
    if (!x.data || !cs.data || !pos.data || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != nh * hd || cs.rank != 2 || cs.shape[1] != rot || numel(&pos) != x.shape[0])
        return RAD_E_SHAPE;
    const int64_t M = x.shape[0], rows = cs.shape[0], half = rot / 2;
    const float sign = inv ? -1.0f : 1.0f;
    AVX_PARALLEL_FOR_IF(M * nh * rot >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t p = ld_at(pos, m);
        if (p < 0 || p >= rows) continue;
        float* c = scratch_f32((size_t)rot);
        const int64_t cb = rowoff(&cs, p, rot), css = laststride(&cs);
        for (int64_t i = 0; i < half; ++i) {
            c[i] = ldt(&cs, cb + i * css);
            c[half + i] = sign * ldt(&cs, cb + (half + i) * css);
        }
        const int64_t xb = rowoff(&x, m, nh * hd), xs = laststride(&x);
        for (int64_t h = 0; h < nh; ++h)
            for (int64_t i = 0; i < half; ++i) {
                const int64_t ia = h * hd + off + (gptj ? 2 * i : i);
                const int64_t ib = gptj ? ia + 1 : ia + half;
                const float co = c[i], si = c[half + i];
                const float va = ldt(&x, xb + ia * xs), vb = ldt(&x, xb + ib * xs);
                stt(&x, xb + ia * xs, va * co - vb * si);
                stt(&x, xb + ib * xs, va * si + vb * co);
            }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ quant_roundtrip */
AVX_KERNEL(quant_roundtrip) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor& x = a->t[0];
    const int64_t n = p_int(a, "n", 0), block = p_int(a, "block", 0);
    const char* fs = p_str(a, "format", "");
    const char* ss = p_str(a, "scale", "");
    QFmt f;
    QScale s;
    if (!std::strcmp(fs, "fp8_e4m3")) f = QFmt::Fp8E4m3;
    else if (!std::strcmp(fs, "fp4_e2m1")) f = QFmt::Fp4E2m1;
    else return RAD_E_UNSUPPORTED;
    if (!std::strcmp(ss, "ue8m0")) s = QScale::Ue8m0;
    else if (!std::strcmp(ss, "e4m3") && f == QFmt::Fp4E2m1) s = QScale::E4m3;
    else if (!std::strcmp(ss, "f32") && f == QFmt::Fp8E4m3) s = QScale::F32;
    else return RAD_E_UNSUPPORTED;
    if (n < 1 || block < 1 || n % block || !x.data || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != n)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(x.shape[0] * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < x.shape[0]; ++m) {
        const int64_t xb = rowoff(&x, m, n), xs = laststride(&x);
        float* v = scratch_f32((size_t)block);
        for (int64_t b0 = 0; b0 < n; b0 += block) {
            float amax = 0;
            for (int64_t i = 0; i < block; ++i) {
                v[i] = ldt(&x, xb + (b0 + i) * xs);
                amax = std::max(amax, std::fabs(v[i]));
            }
            const float sc = block_scale(amax, f, s);
            for (int64_t i = 0; i < block; ++i)
                stt(&x, xb + (b0 + i) * xs, roundtrip(v[i], sc, f));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ softmax_pool */
AVX_KERNEL(softmax_pool) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor &st = a->t[0], &out = a->t[1];
    const RadTensor* page = a->n_t > 2 && a->t[2].data ? &a->t[2] : nullptr;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0);
    if (n < 1 || ratio < 1 || ratio > 64 || !st.data || !out.data || st.rank != 2 ||
        out.rank != 2 || out.shape[0] < 1 || out.shape[1] != n ||
        st.shape[0] != out.shape[0] * ratio || st.shape[1] != 2 * n ||
        (page && numel(page) < out.shape[0]))
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(out.shape[0] * ratio * n >= AVX_PAR_MIN)
    for (int64_t w = 0; w < out.shape[0]; ++w) {
        if (page && ld_at(*page, w) < 0) continue;
        /* The entry's ratio rows, values then scores, widened. */
        float* rowsv = scratch_f32((size_t)(ratio * 2 * n));
        const int64_t ss = laststride(&st);
        for (int64_t i = 0; i < ratio; ++i) {
            const int64_t b = rowoff(&st, w * ratio + i, 2 * n);
            for (int64_t c = 0; c < 2 * n; ++c) rowsv[i * 2 * n + c] = ldt(&st, b + c * ss);
        }
        const int64_t ob = rowoff(&out, w, n), os = laststride(&out);
        for (int64_t c = 0; c < n; ++c) {
            float sc[64], mx = -INFINITY, den = 0, acc = 0;
            for (int64_t i = 0; i < ratio; ++i) {
                sc[i] = rowsv[i * 2 * n + n + c];
                mx = std::max(mx, sc[i]);
            }
            for (int64_t i = 0; i < ratio; ++i) {
                sc[i] = std::exp(sc[i] - mx);
                den += sc[i];
            }
            for (int64_t i = 0; i < ratio; ++i) acc += sc[i] / den * rowsv[i * 2 * n + c];
            stt(&out, ob + c * os, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_sparse_sink
 * libref's addressing and refusals; per row, the keys it reaches are widened once into a
 * [nkeys, D] block (a key is its own value), shared by every head. */
AVX_KERNEL(attn_sparse_sink) {
    if (!a || a->n_t < 11 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wc = a->t[1], &wbt = a->t[2], &wused = a->t[3];
    const RadTensor *cc = a->t[4].data ? &a->t[4] : nullptr, *cbt = a->t[5].data ? &a->t[5] : nullptr;
    const RadTensor* sel = a->t[6].data ? &a->t[6] : nullptr;
    const RadTensor &pos = a->t[7], &cu = a->t[8], &sink = a->t[9], &out = a->t[10];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "dim", 0), win = p_int(a, "window", 0);
    const int64_t ratio = p_int(a, "ratio", 0), topk = p_int(a, "topk", 0);
    const float scale = p_f32(a, "scale", 0.0f);
    const bool comp = ratio > 0;
    const int lf = rad_kvf_parse(p_str(a, "latent_format", "bf16"));
    if (lf < 0) return RAD_E_UNSUPPORTED;
    const int64_t cslots = lf ? rad_kvf_bytes(lf, D) : D;
    if (H < 1 || D < 2 || D > 4096 || win < 1 || ratio < 0 || topk < 0 || !std::isfinite(scale) ||
        scale <= 0 || (comp && (!cc || !cbt || !sel || topk < 1 || D % ratio || cslots < 1 ||
                                cslots % ratio)))
        return RAD_E_SHAPE;
    if (!q.data || !wc.data || !wbt.data || !wused.data || !pos.data || !cu.data || !sink.data ||
        !out.data || q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || out.rank != 2 ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D || wc.rank != 4 || wc.shape[1] != 1 ||
        wc.shape[3] != D || wbt.rank != 2 || numel(&pos) != q.shape[0] || numel(&sink) != H ||
        sink.dtype != RAD_F32 || numel(&cu) < 2 || numel(&wused) < numel(&cu) - 1)
        return RAD_E_SHAPE;
    const int64_t cbs = comp ? cc->shape[2] : 0;
    if (comp && (cc->rank != 4 || cc->shape[1] != 1 || cc->shape[3] * ratio != cslots || cbs % ratio ||
                 cbt->rank != 2 || sel->rank != 2 || sel->shape[0] != q.shape[0] ||
                 sel->shape[1] != topk || cbt->shape[0] != wbt.shape[0]))
        return RAD_E_SHAPE;
    if (comp && lf && (cc->dtype != RAD_F8E4M3 || !rad_tensor_is_contiguous(cc))) return RAD_E_SHAPE;
    const int64_t M = q.shape[0], S = numel(&cu) - 1, bs = wc.shape[2];
    const int64_t nkeys = win + (comp ? topk : 0);
    AVX_PARALLEL_FOR_IF(M * H * nkeys * D >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_of_row(cu, S, m);
        const int64_t p = ld_at(pos, m);
        const int64_t last = ld_at(pos, ld_at(cu, s + 1) - 1);
        const int64_t first = last + 1 - ld_at(wused, s);
        /* scratch: keys [nkeys, D], then scores [nkeys], the accumulator [D], the query head [D] */
        float* keys = scratch_f32((size_t)(nkeys * D + nkeys + 2 * D));
        float* sc = keys + nkeys * D;
        float* acc = sc + nkeys;
        float* qh = acc + D;
        int64_t* live = (int64_t*)scratch_raw((size_t)nkeys * sizeof(int64_t));
        int64_t nl = 0;
        for (int64_t i = 0; i < win; ++i) {
            const int64_t j = p - win + 1 + i, r = j - first;
            if (j < 0 || r < 0 || r / bs >= wbt.shape[1]) continue;
            const int64_t blk = ldi(&wbt, rowoff(&wbt, s, wbt.shape[1]) + (r / bs) * laststride(&wbt));
            if (blk < 0 || blk >= wc.shape[0]) continue;
            const int64_t ko = ((blk * bs) + r % bs) * D;
            float* k = keys + nl * D;
            for (int64_t d = 0; d < D; ++d) k[d] = ldt(&wc, ko + d);
            live[nl++] = i;
        }
        for (int64_t t = 0; comp && t < topk; ++t) {
            const int64_t c = ldi(sel, rowoff(sel, m, topk) + t * laststride(sel));
            if (c < 0 || (c * ratio) / cbs >= cbt->shape[1]) continue;
            const int64_t blk =
                ldi(cbt, rowoff(cbt, s, cbt->shape[1]) + ((c * ratio) / cbs) * laststride(cbt));
            if (blk < 0 || blk >= cc->shape[0]) continue;
            const int64_t ko = (blk * cbs + (c * ratio) % cbs) * (cslots / ratio);
            float* k = keys + nl * D;
            if (lf) {
                const uint8_t* e = (const uint8_t*)cc->data + ko;
                for (int64_t d = 0; d < D; ++d) k[d] = rad_kvf_value(lf, e, D, d);
            } else {
                for (int64_t d = 0; d < D; ++d) k[d] = ldt(cc, ko + d);
            }
            live[nl++] = win + t;
        }
        const int64_t qs = laststride(&q), os = laststride(&out);
        for (int64_t h = 0; h < H; ++h) {
            const int64_t qb = rowoff(&q, m, H * D) + h * D * qs;
            for (int64_t d = 0; d < D; ++d) qh[d] = ldt(&q, qb + d * qs);
            float mx = -1e30f;
            for (int64_t k = 0; k < nl; ++k) {
                sc[k] = dotv(qh, keys + k * D, D) * scale;
                mx = std::max(mx, sc[k]);
            }
            float den = 0;
            std::fill(acc, acc + D, 0.0f);
            for (int64_t k = 0; k < nl; ++k) {
                const float pk = std::exp(sc[k] - mx);
                den += pk;
                axpy(acc, pk, keys + k * D, D);
            }
            den += std::exp(ldt(&sink, h * laststride(&sink)) - mx);
            const int64_t ob = rowoff(&out, m, H * D) + h * D * os;
            for (int64_t d = 0; d < D; ++d) stt(&out, ob + d * os, acc[d] / den);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ index_score
 * libref's rounding points: the dot, w_h, relu(dot) w_h and the head sum each rounded once to
 * q's dtype. */
AVX_KERNEL(index_score) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const int64_t H = p_int(a, "heads", 0), Dm = p_int(a, "dim", 0), ratio = p_int(a, "ratio", 0);
    const int64_t part = p_int(a, "row_part", -1);
    const float wscale = p_f32(a, "wscale", 0.0f);
    const int kf = rad_kvf_parse(p_str(a, "key_format", "bf16"));
    if (kf < 0) return RAD_E_UNSUPPORTED;
    const int64_t kslots = kf ? rad_kvf_bytes(kf, Dm) : Dm;
    if (H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || kslots < 1 || kslots % ratio ||
        !std::isfinite(wscale) || q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * Dm ||
        w.rank != 2 || w.shape[0] != q.shape[0] || w.shape[1] != H || kc.rank != 4 ||
        kc.shape[1] != 1 || kc.shape[3] * ratio != kslots || kc.shape[2] % ratio || bt.rank != 2 ||
        numel(&pos) != q.shape[0] || numel(&cu) < 2 || sc.rank != 2 || sc.shape[0] < 1 ||
        (part < 0 && sc.shape[0] != q.shape[0]) || sc.shape[1] < 1 || sc.dtype != RAD_F32)
        return RAD_E_SHAPE;
    if (kf && (kc.dtype != RAD_F8E4M3 || !rad_tensor_is_contiguous(&kc))) return RAD_E_SHAPE;
    const int64_t Mq = q.shape[0], M = sc.shape[0], C = sc.shape[1], S = numel(&cu) - 1;
    const int64_t cbs = kc.shape[2];
    const int64_t row0 = part > 0 ? part * M : 0;
    const uint32_t dt = q.dtype;
    AVX_PARALLEL_FOR_IF(M * C * H * Dm >= AVX_PAR_MIN)
    for (int64_t i = 0; i < M; ++i) {
        const int64_t m = row0 + i;
        const int64_t sb = rowoff(&sc, i, C), ss = laststride(&sc);
        if (m >= Mq) {
            for (int64_t j = 0; j < C; ++j) stt(&sc, sb + j * ss, -INFINITY);
            continue;
        }
        const int64_t s = seq_of_row(cu, S, m);
        const int64_t reach = (ld_at(pos, m) + 1) / ratio;
        /* scratch: the query row [H Dm], one key [Dm], the head weights [H] */
        float* qv = scratch_f32((size_t)(H * Dm + Dm + H));
        float* kv = qv + H * Dm;
        float* wh = kv + Dm;
        const int64_t qb = rowoff(&q, m, H * Dm), qs = laststride(&q);
        for (int64_t d = 0; d < H * Dm; ++d) qv[d] = ldt(&q, qb + d * qs);
        for (int64_t h = 0; h < H; ++h)
            wh[h] = narrow1(dt, ldt(&w, rowoff(&w, m, H) + h * laststride(&w)) * wscale);
        for (int64_t j = 0; j < C; ++j) {
            float v = -INFINITY;
            const int64_t col = (j * ratio) / cbs;
            if (j < reach && col < bt.shape[1]) {
                const int64_t blk = ldi(&bt, rowoff(&bt, s, bt.shape[1]) + col * laststride(&bt));
                if (blk >= 0 && blk < kc.shape[0]) {
                    const int64_t ko = (blk * cbs + (j * ratio) % cbs) * (kslots / ratio);
                    if (kf) {
                        const uint8_t* e = (const uint8_t*)kc.data + ko;
                        for (int64_t d = 0; d < Dm; ++d) kv[d] = rad_kvf_value(kf, e, Dm, d);
                    } else {
                        for (int64_t d = 0; d < Dm; ++d) kv[d] = ldt(&kc, ko + d);
                    }
                    float acc = 0;
                    for (int64_t h = 0; h < H; ++h) {
                        const float dot = narrow1(dt, dotv(qv + h * Dm, kv, Dm));
                        acc += narrow1(dt, std::max(dot, 0.0f) * wh[h]);
                    }
                    v = narrow1(dt, acc);
                }
            }
            stt(&sc, sb + j * ss, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ index_candidates */
AVX_KERNEL(index_candidates) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &keep = a->t[2];
    const int64_t block = p_int(a, "block", 0), kb = p_int(a, "topk_blocks", 0);
    const int64_t ratio = p_int(a, "ratio", 0);
    if (block < 1 || kb < 1 || ratio < 1 || sc.rank != 2 || sc.shape[0] < 1 || sc.shape[1] < 1 ||
        sc.dtype != RAD_F32 || numel(&pos) != sc.shape[0] || keep.rank != 2 ||
        keep.shape[0] != sc.shape[0] || keep.shape[1] != keep_words(sc.shape[1], block))
        return RAD_E_SHAPE;
    if (keep.dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t M = sc.shape[0], C = sc.shape[1], nb = (C + block - 1) / block;
    const int64_t KW = keep.shape[1];
    AVX_PARALLEL_FOR_IF(M * C >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> bsc((size_t)nb, -INFINITY);
        const int64_t sb = rowoff(&sc, m, C), ss = laststride(&sc);
        for (int64_t j = 0; j < C; ++j)
            bsc[(size_t)(j / block)] = std::max(bsc[(size_t)(j / block)], ldt(&sc, sb + j * ss));
        const int64_t reach = (ld_at(pos, m) + 1) / ratio;
        if (reach >= 1 && (reach - 1) / block < nb) bsc[(size_t)((reach - 1) / block)] = INFINITY;
        std::vector<uint32_t> bits((size_t)KW, 0u);
        for (int64_t b : top_indices(bsc, kb))
            if (bsc[(size_t)b] > -INFINITY) bits[(size_t)(b / 32)] |= 1u << (b % 32);
        for (int64_t wd = 0; wd < KW; ++wd)
            st_int_dt(keep.dtype, keep.data, rowoff(&keep, m, KW) + wd * laststride(&keep),
                      (int32_t)bits[(size_t)wd]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ index_topk */
AVX_KERNEL(index_topk) {
    if (!a || a->n_t < 4 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[2], &sel = a->t[3];
    const RadTensor* keep = a->t[1].data ? &a->t[1] : nullptr;
    const int64_t topk = p_int(a, "topk", 0), ratio = p_int(a, "ratio", 0);
    const int64_t block = p_int(a, "block", 0);
    if (topk < 1 || ratio < 1 || !sc.data || !pos.data || !sel.data || sc.rank != 2 ||
        sc.shape[0] < 1 || sc.dtype != RAD_F32 || numel(&pos) != sc.shape[0] || sel.rank != 2 ||
        sel.shape[0] != sc.shape[0] || sel.shape[1] != topk ||
        (keep && (block < 1 || keep->rank != 2 || keep->shape[0] != sc.shape[0] ||
                  keep->shape[1] != keep_words(sc.shape[1], block))))
        return RAD_E_SHAPE;
    if (keep && keep->dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t M = sc.shape[0], C = sc.shape[1], KW = keep ? keep->shape[1] : 0;
    AVX_PARALLEL_FOR_IF(M * C >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t reach = std::min<int64_t>(C, (ld_at(pos, m) + 1) / ratio);
        std::vector<float> v((size_t)std::max<int64_t>(reach, 0), -INFINITY);
        const int64_t sb = rowoff(&sc, m, C), ss = laststride(&sc);
        for (int64_t j = 0; j < reach; ++j) {
            if (keep) {
                const int64_t b = j / block;
                const uint32_t wd =
                    (uint32_t)ldi(keep, rowoff(keep, m, KW) + (b / 32) * laststride(keep));
                if (!((wd >> (b % 32)) & 1u)) continue;
            }
            v[(size_t)j] = ldt(&sc, sb + j * ss);
        }
        std::vector<int64_t> pick;
        for (int64_t j : top_indices(v, topk))
            if (v[(size_t)j] > -INFINITY) pick.push_back(j);
        std::sort(pick.begin(), pick.end());
        for (int64_t t = 0; t < topk; ++t)
            st_int_dt(sel.dtype, sel.data, rowoff(&sel, m, topk) + t * laststride(&sel),
                      t < (int64_t)pick.size() ? pick[(size_t)t] : -1);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ entry_store
 * Every page is checked before any entry is written; libref writes the entries before a page
 * past the cache and then refuses with the same code. An entry is a few hundred values, so the
 * rows run serially. */
AVX_KERNEL(entry_store) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const int64_t n = p_int(a, "n", 0), bs = p_int(a, "block_size", 0);
    const int f = rad_kvf_parse(p_str(a, "format", "bf16"));
    if (f < 0) return RAD_E_UNSUPPORTED;
    const int64_t ne = f ? rad_kvf_bytes(f, n) : n;
    if (n < 1 || ne < 1 || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] != ne)
        return RAD_E_SHAPE;
    if (f && (cache.dtype != RAD_F8E4M3 || !rad_tensor_is_contiguous(&cache))) return RAD_E_SHAPE;
    for (int64_t w = 0; w < x.shape[0]; ++w)
        if (ld_at(page, w) >= cache.shape[0]) return RAD_E_INVAL;
    /* Serial, as libref: two rows naming one page leave the later row's entry. */
    for (int64_t w = 0; w < x.shape[0]; ++w) {
        const int64_t pg = ld_at(page, w);
        if (pg < 0) continue;
        const int64_t xb = rowoff(&x, w, n), xs = laststride(&x);
        if (f) {
            float* row = scratch_f32((size_t)n);
            for (int64_t i = 0; i < n; ++i) row[i] = ldt(&x, xb + i * xs);
            rad_kvf_pack(f, row, n, (uint8_t*)cache.data + pg * ne);
            continue;
        }
        for (int64_t i = 0; i < n; ++i) stt(&cache, pg * n + i, ldt(&x, xb + i * xs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ token_map */
AVX_KERNEL(token_map) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &tab = a->t[1], &y = a->t[2];
    const int64_t nv = p_int(a, "n_vocab", 0);
    if (nv < 1 || numel(&tab) != nv || numel(&x) < 1 || numel(&y) != numel(&x)) return RAD_E_SHAPE;
    for (int64_t m = 0; m < numel(&x); ++m) {
        const int64_t id = ld_at(x, m);
        const int64_t v = id < 0 || id >= nv ? -1 : ld_at(tab, id);
        st_int_dt(y.dtype, y.data, offlin(&y, m), v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ stream_gate_add */
AVX_KERNEL(stream_gate_add) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor &h = a->t[0], &k = a->t[1], &v = a->t[2], &w = a->t[3], &out = a->t[4];
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", -1.0f), ge = p_f32(a, "gate_eps", -1.0f);
    if (n < 1 || hc < 1 || hc > 16 || !(eps >= 0) || !(ge >= 0) || h.rank != 2 || h.shape[0] < 1 ||
        h.shape[1] != hc * n || k.rank != 2 || k.shape[0] != h.shape[0] || k.shape[1] != hc * n ||
        v.rank != 2 || v.shape[0] != h.shape[0] || v.shape[1] != n || numel(&w) != hc * n ||
        w.dtype != RAD_F32 || out.rank != 2 || out.shape[0] != h.shape[0] ||
        out.shape[1] != hc * n)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(h.shape[0] * hc * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < h.shape[0]; ++m) {
        /* scratch: h's row, k's row, h * w * k, v's row */
        float* hv = scratch_f32((size_t)(3 * hc * n + n));
        float* kv = hv + hc * n;
        float* hw = kv + hc * n;
        float* vv = hw + hc * n;
        const int64_t hb = rowoff(&h, m, hc * n), kb = rowoff(&k, m, hc * n);
        const int64_t vb = rowoff(&v, m, n), ob = rowoff(&out, m, hc * n);
        for (int64_t i = 0; i < hc * n; ++i) {
            hv[i] = ldt(&h, hb + i * laststride(&h));
            kv[i] = ldt(&k, kb + i * laststride(&k));
            hw[i] = hv[i] * ldt(&w, i * laststride(&w));
        }
        for (int64_t i = 0; i < n; ++i) vv[i] = ldt(&v, vb + i * laststride(&v));
        for (int64_t s = 0; s < hc; ++s) {
            const float* hs = hv + s * n;
            const float* ks = kv + s * n;
            const float sh = dotv(hs, hs, n), sk = dotv(ks, ks, n);
            float dot = dotv(hw + s * n, ks, n);
            const float rstd = (1.0f / std::sqrt(sh / (float)n + eps)) *
                               (1.0f / std::sqrt(sk / (float)n + eps));
            dot = dot * rstd * (1.0f / std::sqrt((float)n));
            const float gate =
                1.0f / (1.0f + std::exp(-std::copysign(std::sqrt(std::max(std::fabs(dot), ge)), dot)));
            for (int64_t i = 0; i < n; ++i)
                stt(&out, ob + (s * n + i) * laststride(&out), hs[i] + gate * vv[i]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ embed_lookup_planes */
AVX_KERNEL(embed_lookup_planes) {
    if (a->n_t < 5 || !rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 2) ||
        !rad_arg_in(a, 4))
        return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &cd = a->t[1], &sc = a->t[2], &tb = a->t[3], &x = a->t[4];
    const bool tab = tb.data != nullptr;
    const int64_t E = p_int(a, "n_embd", 0), V = p_int(a, "n_vocab", 0), G = p_int(a, "group", 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (E <= 0 || V <= 0 || G <= 0 || E % G || voff < 0 || cd.rank != 2 || cd.shape[0] < V ||
        cd.shape[1] != E || sc.rank != 2 || sc.shape[0] < V || sc.shape[1] != E / G)
        return RAD_E_SHAPE;
    const uint32_t cdt = cd.dtype;
    const int bits = (cdt == RAD_U4 || cdt == RAD_I4) ? 4 : (cdt == RAD_U8 || cdt == RAD_I8) ? 8 : 0;
    if (!bits) return RAD_E_DTYPE;
    if (tab && (cdt == RAD_I4 || cdt == RAD_I8 || numel(&tb) < (int64_t)1 << bits)) return RAD_E_DTYPE;
    const int64_t rb = rad_dtype_bytes(cdt, E);
    const int64_t M = numel(&x) / E;
    if (rb <= 0 || M <= 0 || numel(&tok) < M) return RAD_E_SHAPE;
    if (voff == 0 && a->world_size <= 1)
        for (int64_t m = 0; m < M; ++m)
            if (ld_at(tok, m) >= V) return RAD_E_INVAL;
    /* The table, widened once: at most 256 codes. */
    float lut[256];
    if (tab)
        for (int c = 0; c < (1 << bits); ++c) lut[c] = ld_elem(tb.data, tb.dtype, c);
    AVX_PARALLEL_FOR_IF(M * E >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ld_at(tok, m) - voff;
        const int64_t xb = rowoff(&x, m, E), xs = laststride(&x);
        if (id < 0 || id >= V) {
            for (int64_t i = 0; i < E; ++i) stt(&x, xb + i * xs, 0.0f);
            continue;
        }
        const uint8_t* row = (const uint8_t*)cd.data + id * rb;
        const int64_t sb = rowoff(&sc, id, E / G), ss = laststride(&sc);
        for (int64_t g = 0; g < E / G; ++g) {
            const float s = ldt(&sc, sb + g * ss);
            for (int64_t i = g * G; i < (g + 1) * G; ++i) {
                int code;
                if (bits == 8) {
                    code = cdt == RAD_I8 ? (int)(int8_t)row[i] : (int)row[i];
                } else {
                    const int u = (row[i >> 1] >> ((i & 1) * 4)) & 15;
                    code = cdt == RAD_I4 ? (u ^ 8) - 8 : u;
                }
                const float v = tab ? lut[code] : (float)code;
                stt(&x, xb + i * xs, v * s * mult);
            }
        }
    }
    return RAD_OK;
}
