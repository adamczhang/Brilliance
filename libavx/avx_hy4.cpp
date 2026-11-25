/* avx_hy4.cpp -- Tencent Hy4-flash's new ops on the host: independent hyper-connections, absorbed
 * and non-absorbed attention with a sink, the compressed indexer's pooling and score, gate_mul and
 * Engram (docs/OPS.md, "Hy4-flash"; the semantics are libref/ref_hy4.cpp's).
 *
 * Conventions, as in the rest of the plugin:
 *   - every extent comes from an operand, never from a ranged parameter, and addressing is libref's
 *     (rowoff / laststride), so strided and rank-2 head views work;
 *   - rows are widened to f32 once (in_row), computed, and narrowed once (out_row);
 *   - dot products are vectorised (dotf), so they match libref to rounding, not in bytes. Where
 *     the note marks an order (idx_score's two head halves, Engram's f64 reductions, iHC's
 *     ascending stream sum), the order is kept;
 *   - one thread owns whole outputs, no reduction clause, so the bytes do not depend on the thread
 *     count (the checker reruns every case on several threads and compares).
 *
 * Attention is one (query row, head) at a time with an online softmax that starts at the sink:
 * `max = sink`, `den = 0`, and the sink joins the denominator once at the end (exp(sink - max)),
 * which is libref's "maximum over the scores and the sink". No visible key writes zero.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace avx;

/* A dynamic schedule for the attention rows: a causal prefill's late rows cost far more than its
 * early ones, so contiguous blocks would leave one thread with most of the work. A unit's
 * arithmetic does not depend on the thread that runs it. */
#ifdef _OPENMP
#define HY4_PAR_DYN(work) \
    AVX_PRAGMA(omp parallel for schedule(dynamic, 1) if((work) >= ((int64_t)1 << 19) || \
               avx_par_forced()) num_threads(avx_team_size()))
#else
#define HY4_PAR_DYN(work)
#endif

namespace {

/* [rows, heads*width] read as [rows, heads, width] when it is declared the narrow way. */
void heads_view(RadTensor& t, int64_t h, int64_t width) {
    if (h > 0 && width > 0 && t.rank == 2 && t.shape[1] == h * width) {
        t.rank = 3;
        t.shape[1] = h;
        t.shape[2] = width;
        t.stride[2] = t.stride[1];
        t.stride[1] *= width;
    }
}

bool matrix(const RadTensor& t, int64_t rows, int64_t cols) {
    return t.data && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}

int64_t cuv(const RadTensor* cu, int64_t i) { return ld_int_dt(cu->dtype, cu->data, offlin(cu, i)); }

int64_t seq_row(const RadTensor* cu, int64_t n_seq, int64_t r) {
    int64_t s = 0;
    while (s + 1 < n_seq && cuv(cu, s + 1) <= r) ++s;
    return s;
}

/* floor modulus (Python's), and torch's wrapping int64 product. */
int64_t floor_mod(int64_t a, int64_t b) {
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}
int64_t wrap_mul(int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); }

float sign_of(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

/* 1 / (1 + e^-x) with libref's double exponential. */
float sigmoid1(float x) { return 1.0f / (1.0f + sc_exp(-x)); }

/* v[0, n) rounded through dtype `dt` in place; `tmp` holds n 16-bit words. */
void round_through(uint32_t dt, float* v, int64_t n, uint16_t* tmp) {
    if (dt == RAD_F32) return;
    if (dt == RAD_BF16) {
        cvt_f32_to_bf16(v, tmp, n);
        cvt_bf16_to_f32(tmp, v, n);
        return;
    }
    for (int64_t i = 0; i < n; ++i) v[i] = narrow1(dt, v[i]);
}

}  // namespace

/* ================================================================== ihc_pre / ihc_post */

/* f = f32(h); r = rsqrt(sum f^2 / (hc n) + eps); c[j] = (fn[j] . f) r
 * pre[j] = sigmoid(c[j] scale[0] + base[j]) + hc_eps; x = round(sum_j pre[j] f[j]) (j ascending)
 * post[j] = post_mag sigmoid(c[hc+j] scale[1] + base[hc+j]) + hc_eps, with `post`. */
AVX_KERNEL(ihc_pre) {
    if (a->n_t < 5 || !rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3], &x = a->t[4];
    const RadTensor* post = a->n_t > 5 ? rad_arg_in(a, 5) : nullptr;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0), rows = p_int(a, "rows", 0);
    const float eps = p_f32(a, "eps", 0), heps = p_f32(a, "hc_eps", 0);
    const int64_t width = hc * n;
    if (n < 1 || hc < 1 || hc > 16 || rows != (post ? 2 * hc : hc) || !std::isfinite(eps) ||
        eps <= 0 || !std::isfinite(heps) || heps < 0)
        return RAD_E_SHAPE;
    float pmag = 0;
    if (post) {
        if (!p_find(a, "post_mag")) return RAD_E_INVAL;
        pmag = p_f32(a, "post_mag", 0);
        if (!std::isfinite(pmag)) return RAD_E_SHAPE;
    }
    if (h.rank != 2 || h.shape[0] < 1) return RAD_E_SHAPE;
    const int64_t M = h.shape[0];
    if (!matrix(h, M, width) || !matrix(fn, rows, width) || scale.rank != 1 ||
        numel(&scale) != (post ? 2 : 1) || base.rank != 1 || numel(&base) != rows ||
        !matrix(x, M, n) || (post && !matrix(*post, M, hc)))
        return RAD_E_SHAPE;
    if (post && post->dtype != RAD_F32) return RAD_E_DTYPE;
    const float s0 = ldt(&scale, 0), s1 = post ? ldt(&scale, laststride(&scale)) : 0.0f;
    float bs[32];
    for (int64_t j = 0; j < rows; ++j) bs[j] = ldt(&base, j * laststride(&base));

    AVX_PARALLEL_FOR_IF(M * (rows + 1) * width >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        float* tmp = scratch_f32((size_t)(2 * width + n + 16));
        float *fbuf = tmp, *wbuf = tmp + width, *xbuf = tmp + 2 * width;
        const float* f = in_row(&h, rowoff(&h, r, width), width, fbuf);
        const float ms = dotf(f, f, width) / (float)width;
        const float inv = 1.0f / std::sqrt(ms + eps);
        float c[32], pre[16];
        for (int64_t j = 0; j < rows; ++j) {
            const float* w = in_row(&fn, rowoff(&fn, j, width), width, wbuf);
            c[j] = dotf(w, f, width) * inv;
        }
        for (int64_t j = 0; j < hc; ++j) pre[j] = sigmoid1(c[j] * s0 + bs[j]) + heps;
        if (post) {
            float pv[16];
            for (int64_t j = 0; j < hc; ++j) pv[j] = pmag * sigmoid1(c[hc + j] * s1 + bs[hc + j]) + heps;
            const int64_t pb = rowoff(post, r, hc), ps = laststride(post);
            for (int64_t j = 0; j < hc; ++j) stt(post, pb + j * ps, pv[j]);
        }
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf acc = vf_zero();
            for (int64_t j = 0; j < hc; ++j)
                acc = vf_fma(vf_set1(pre[j]), vf_loadu(f + j * n + i), acc);
            vf_storeu(xbuf + i, acc);
        }
        for (; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j) v += pre[j] * f[j * n + i];
            xbuf[i] = v;
        }
        out_row(&x, rowoff(&x, r, n), n, xbuf);
    }
    return RAD_OK;
}

/* h[j][i] = round(post[j] * f32(y[i]) + f32(h[j][i])), in place. */
AVX_KERNEL(ihc_post) {
    if (a->n_t < 3 || !rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &y = a->t[0], &post = a->t[1], &h = a->t[2];
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n < 1 || hc < 1 || h.rank != 2 || h.shape[0] < 1) return RAD_E_SHAPE;
    const int64_t M = h.shape[0];
    if (!matrix(h, M, hc * n) || !matrix(y, M, n) || !matrix(post, M, hc)) return RAD_E_SHAPE;
    if (post.dtype != RAD_F32) return RAD_E_DTYPE;
    AVX_PARALLEL_FOR_IF(M * hc * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        float* tmp = scratch_f32((size_t)(2 * n + 16));
        float *ybuf = tmp, *hbuf = tmp + n;
        const float* yv = in_row(&y, rowoff(&y, r, n), n, ybuf);
        const int64_t hb = rowoff(&h, r, hc * n);
        for (int64_t j = 0; j < hc; ++j) {
            const float p = ldt(&post, rowoff(&post, r, hc) + j * laststride(&post));
            const float* hv = in_row(&h, hb + j * n * laststride(&h), n, hbuf);
            const vf pv = vf_set1(p);
            int64_t i = 0;
            /* The product is rounded to f32 before the add. */
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(hbuf + i, vf_add(vf_mul(pv, vf_loadu(yv + i)), vf_loadu(hv + i)));
            for (; i < n; ++i) hbuf[i] = p * yv[i] + hv[i];
            out_row(&h, hb + j * n * laststride(&h), n, hbuf);
        }
    }
    return RAD_OK;
}

/* ================================================================== rmsnorm_ihc_junction */

/* `[add] -> rmsnorm -> ihc_post -> [ihc_pre -> rmsnorm]`: this level's own add, rmsnorm, ihc_post
 * and ihc_pre in sequence (docs/OPS.md), so it is their bytes; `y` and `x` held in temporaries of
 * the activation dtype where the caller passed none. libref's ref_rmsnorm_ihc_junction is the same
 * composition of libref's ops. */
extern "C" int AVX_FN(avx_add)(const RadArgs*, RadStream);
extern "C" int AVX_FN(avx_rmsnorm)(const RadArgs*, RadStream);
extern "C" int AVX_FN(avx_swiglu_clamped)(const RadArgs*, RadStream);
extern "C" int AVX_FN(avx_had_quant_act_i8g)(const RadArgs*, RadStream);
AVX_KERNEL(rmsnorm_ihc_junction) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    auto opd = [&](int i) -> const RadTensor* { return i < a->n_t ? rad_arg_in(a, i) : nullptr; };
    const RadTensor *A = opd(0), *B = opd(1), *wp = opd(2), *post = opd(3), *h = opd(4);
    const RadTensor *fn = opd(5), *sc = opd(6), *bs = opd(7), *wn = opd(8), *xn = opd(9);
    const RadTensor *Y = opd(10), *X = opd(11);
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0), rows = p_int(a, "rows", 0),
                  tail = p_int(a, "tail", 1);
    if (!A || !wp || !post || !h) return RAD_E_INVAL;
    if (tail != 0 && tail != 1) return RAD_E_INVAL;
    if (tail && (!fn || !sc || !bs || !wn || !xn)) return RAD_E_INVAL;
    if (n < 1 || hc < 1 || A->rank != 2 || A->shape[0] < 1 || A->shape[1] != n) return RAD_E_SHAPE;
    const int64_t M = A->shape[0];
    if (tail && rows != hc && rows != 2 * hc) return RAD_E_SHAPE;
    const RadParam* eps = p_find(a, "eps");
    if (!eps) return RAD_E_INVAL;
    const size_t es = A->dtype == RAD_F32 ? 4 : 2;
    std::vector<unsigned char> ty, tx;
    RadTensor yt{}, xt{};
    auto tmp = [&](std::vector<unsigned char>& buf, RadTensor& t) {
        buf.assign((size_t)(M * n) * es, 0);
        t = RadTensor{};
        t.data = buf.data();
        t.dtype = A->dtype;
        t.rank = 2;
        t.shape[0] = M;
        t.shape[1] = n;
        t.stride[0] = n;
        t.stride[1] = 1;
    };
    if (!Y) { tmp(ty, yt); Y = &yt; }
    if (tail && !X) { tmp(tx, xt); X = &xt; }
    RadParam pm[8]{};
    auto P = [&](int k, const char* key, long long v) { pm[k] = RadParam{}; pm[k].key = key; pm[k].kind = RAD_P_INT; pm[k].ival = v; };
    auto PF = [&](int k, const char* key, double v) { pm[k] = RadParam{}; pm[k].key = key; pm[k].kind = RAD_P_F64; pm[k].dval = v; };
    auto run = [&](int (*fnp)(const RadArgs*, RadStream), std::initializer_list<const RadTensor*> ts,
                   int np) {
        RadTensor tv[8]{};
        int k = 0;
        for (const RadTensor* t : ts) tv[k++] = t ? *t : RadTensor{};
        RadArgs sa{};
        sa.t = tv;
        sa.n_t = k;
        sa.p = pm;
        sa.n_p = np;
        sa.world_size = 1;
        return fnp(&sa, stream);
    };
    const double e = eps->kind == RAD_P_F64 ? eps->dval : (double)p_f32(a, "eps", 1e-6f);
    int rc;
    P(0, "M", M);
    P(1, "n", n);
    if (B && (rc = run(AVX_FN(avx_add), { A, B, A }, 2)) != RAD_OK) return rc;
    PF(2, "eps", e);
    if ((rc = run(AVX_FN(avx_rmsnorm), { A, wp, Y }, 3)) != RAD_OK) return rc;
    P(2, "hc", hc);
    if ((rc = run(AVX_FN(avx_ihc_post), { Y, post, h }, 3)) != RAD_OK) return rc;
    if (!tail) return RAD_OK;
    const bool has_post = rows == 2 * hc;
    P(3, "rows", rows);
    PF(4, "eps", (double)p_f32(a, "ihc_eps", 0.0f));
    PF(5, "hc_eps", (double)p_f32(a, "hc_eps", 0.0f));
    int np = 6;
    if (has_post) {
        if (!p_find(a, "post_mag")) return RAD_E_INVAL;
        PF(6, "post_mag", (double)p_f32(a, "post_mag", 0.0f));
        np = 7;
    }
    if ((rc = run(AVX_FN(avx_ihc_pre), { h, fn, sc, bs, X, has_post ? post : nullptr }, np)) != RAD_OK)
        return rc;
    P(0, "M", M);
    P(1, "n", n);
    PF(2, "eps", e);
    if ((rc = run(AVX_FN(avx_rmsnorm), { X, wn, xn }, 3)) != RAD_OK) return rc;
    /* G-3b: the codes of the next block's input norm, had_quant_act_i8g over xn */
    const RadTensor *xq = opd(12), *xs = opd(13);
    if (!xq) return RAD_OK;
    if (!xs || p_int(a, "group", 128) != 128) return RAD_E_INVAL;
    P(2, "group", 128);
    pm[3] = RadParam{}; pm[3].key = "dtype"; pm[3].kind = RAD_P_STR; pm[3].sval = p_str(a, "dtype", "bf16");
    return run(AVX_FN(avx_had_quant_act_i8g), { xn, xq, xs }, 4);
}

/* ================================================================== mla_project_kv */

/* The checkpoint's kv_b per head: head h's rows start at h (k_dim + v_dim), key rows first.
 *   absorb: y[m,h,j] = sum_{c<kd} a[m,h,c] w[h(kd+vd) + c][j]
 *   expand: y[m,h,c] = sum_{j<L}  a[m,h,j] w[h(kd+vd) + kd + c][j]
 * Jobs run head-major, so the threads' neighbouring rows share a head's weights in the cache. */
AVX_KERNEL(mla_project_kv) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    RadTensor x = a->t[0], y = a->t[2];
    const RadTensor& w = a->t[1];
    const int64_t H = p_int(a, "heads", 0), kd = p_int(a, "k_dim", 0), vd = p_int(a, "v_dim", 0),
                  L = p_int(a, "latent_dim", 0);
    const char* mode = p_str(a, "mode", "");
    const bool absorb = !std::strcmp(mode, "absorb"), expand = !std::strcmp(mode, "expand");
    if ((!absorb && !expand) || H < 1 || kd < 1 || vd < 1 || L < 1) return RAD_E_SHAPE;
    const int64_t in_w = absorb ? kd : L, out_w = absorb ? L : vd;
    heads_view(x, H, in_w);
    heads_view(y, H, out_w);
    if (x.rank != 3 || y.rank != 3 || x.shape[0] < 1 || x.shape[1] != H || x.shape[2] != in_w ||
        w.rank != 2 || w.shape[0] != H * (kd + vd) || w.shape[1] != L ||
        y.shape[0] != x.shape[0] || y.shape[1] != H || y.shape[2] != out_w)
        return RAD_E_SHAPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    const int64_t M = x.shape[0];
    AVX_PARALLEL_FOR_IF(M * H * out_w * in_w >= AVX_PAR_MIN)
    for (int64_t job = 0; job < M * H; ++job) {
        const int64_t head = job / M, r = job % M;
        float* tmp = scratch_f32((size_t)(in_w + 2 * L + out_w + 16));
        float *qbuf = tmp, *wbuf = tmp + in_w, *acc = wbuf + L, *obuf = acc + L;
        const float* q = in_row(&x, r * x.stride[0] + head * x.stride[1], in_w, qbuf);
        const int64_t ob = r * y.stride[0] + head * y.stride[1];
        if (absorb) {
            /* acc[j] = sum_c q[c] w[c][j], c ascending, vectorised over j. */
            for (int64_t j = 0; j < L; ++j) acc[j] = 0;
            for (int64_t c = 0; c < kd; ++c) {
                const float* wr = in_row(&w, (head * (kd + vd) + c) * w.stride[0], L, wbuf);
                const vf qv = vf_set1(q[c]);
                int64_t j = 0;
                for (; j + VF_N <= L; j += VF_N)
                    vf_storeu(acc + j, vf_fma(qv, vf_loadu(wr + j), vf_loadu(acc + j)));
                for (; j < L; ++j) acc[j] += q[c] * wr[j];
            }
            out_row(&y, ob, L, acc);
        } else {
            for (int64_t c = 0; c < vd; ++c) {
                const float* wr = in_row(&w, (head * (kd + vd) + kd + c) * w.stride[0], L, wbuf);
                obuf[c] = dotf(q, wr, L);
            }
            out_row(&y, ob, vd, obuf);
        }
    }
    return RAD_OK;
}

/* ================================================================== latent attention with a sink */

namespace {

struct MlaSink {
    const RadTensor *q, *qr, *kv, *bt, *o;
    int64_t L, R, bs;
    float scale;
};

/* One (query row r, head): keys [lo, hi) read through row `bt_row` of the page table; the sink
 * joins the softmax once, the value is the latent. No mapped key in range: zero. */
void mla_sink_one(const MlaSink& c, int64_t r, int64_t head, int64_t bt_row, int64_t lo,
                  int64_t hi, float sink) {
    const int64_t L = c.L, R = c.R, bs = c.bs;
    float* tmp = scratch_f32((size_t)(3 * L + 2 * R + 16));
    float *qb = tmp, *kb = qb + L, *value = kb + L, *qrb = value + L, *rb = qrb + R;
    const float* q = in_row(c.q, r * c.q->stride[0] + head * c.q->stride[1], L, qb);
    const float* qr = in_row(c.qr, r * c.qr->stride[0] + head * c.qr->stride[1], R, qrb);
    for (int64_t i = 0; i < L; ++i) value[i] = 0;
    float mx = sink, den = 0;
    bool any = false;
    const RadTensor &kv = *c.kv, &bt = *c.bt;
    for (int64_t t = lo; t < hi; ++t) {
        const int64_t page = ld_int_dt(bt.dtype, bt.data, bt_row * bt.stride[0] + (t / bs) * bt.stride[1]);
        if (page < 0 || page >= kv.shape[0]) continue;
        const int64_t b = page * kv.stride[0] + (t % bs) * kv.stride[2];
        const float* key = in_row(&kv, b, L, kb);
        const float* rot = in_row(&kv, b + L * kv.stride[3], R, rb);
        const float s = (dotf(q, key, L) + dotf(qr, rot, R)) * c.scale;
        float w;
        if (s > mx) {
            const float old = std::exp(mx - s);
            const vf ov = vf_set1(old);
            int64_t i = 0;
            for (; i + VF_N <= L; i += VF_N) vf_storeu(value + i, vf_mul(vf_loadu(value + i), ov));
            for (; i < L; ++i) value[i] *= old;
            den *= old;
            mx = s;
            w = 1.0f;
        } else {
            w = std::exp(s - mx);
        }
        den += w;
        const vf wv = vf_set1(w);
        int64_t i = 0;
        for (; i + VF_N <= L; i += VF_N)
            vf_storeu(value + i, vf_fma(vf_loadu(key + i), wv, vf_loadu(value + i)));
        for (; i < L; ++i) value[i] += w * key[i];
        any = true;
    }
    if (!any) {
        for (int64_t i = 0; i < L; ++i) value[i] = 0;
    } else {
        den += std::exp(sink - mx);
        const vf dv = vf_set1(den);
        int64_t i = 0;
        for (; i + VF_N <= L; i += VF_N) vf_storeu(value + i, vf_div(vf_loadu(value + i), dv));
        for (; i < L; ++i) value[i] /= den;
    }
    out_row(c.o, r * c.o->stride[0] + head * c.o->stride[1], L, value);
}

bool sinks_ok(const RadTensor& s, int64_t H) { return s.rank == 1 && numel(&s) >= H; }

}  // namespace

/* mla_attn_paged with a window and a sink. Sequence s owns rows [cu[s], cu[s+1]); its context after
 * the step is seqused[s]; row i sits at p = seqused[s] - qlen + i and sees [max(0, p - window + 1),
 * p] (window 0: from 0). */
AVX_KERNEL(mla_attn_paged_sink) {
    if (a->n_t < 8 || !rad_args_have(a, 8)) return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[7];
    const RadTensor &kv = a->t[2], &bt = a->t[3], &su = a->t[4], &cu = a->t[5], &sinks = a->t[6];
    const int64_t H = p_int(a, "heads", 0), L = p_int(a, "latent_dim", 0),
                  R = p_int(a, "rope_dim", 0), bs = p_int(a, "block_size", 0),
                  window = p_int(a, "window", 0);
    const float scale = p_f32(a, "scale", 0);
    heads_view(q, H, L);
    heads_view(qr, H, R);
    heads_view(o, H, L);
    if (H < 1 || L < 1 || R < 1 || bs < 1 || window < 0 || !std::isfinite(scale) || scale <= 0 ||
        q.rank != 3 || q.shape[0] < 1 || q.shape[1] != H || q.shape[2] != L || qr.rank != 3 ||
        qr.shape[0] != q.shape[0] || qr.shape[1] != H || qr.shape[2] != R || kv.rank != 4 ||
        kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < L + R || bt.rank != 2 ||
        su.rank != 1 || cu.rank != 1 || numel(&cu) != numel(&su) + 1 || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != H || o.shape[2] != L || !sinks_ok(sinks, H))
        return RAD_E_SHAPE;
    const int64_t n_seq = numel(&su), rows = q.shape[0];
    if (cuv(&cu, 0) != 0 || cuv(&cu, n_seq) != rows || bt.shape[0] < n_seq) return RAD_E_SHAPE;
    const MlaSink c{ &q, &qr, &kv, &bt, &o, L, R, bs, scale };
    const int64_t ctx_max = bt.shape[1] * bs;
    HY4_PAR_DYN(rows * H * (L + R) * std::min<int64_t>(ctx_max, 4096))
    for (int64_t job = 0; job < rows * H; ++job) {
        const int64_t r = job / H, head = job % H;
        const int64_t s = seq_row(&cu, n_seq, r);
        const int64_t qlen = cuv(&cu, s + 1) - cuv(&cu, s), i = r - cuv(&cu, s);
        const int64_t pos = ld_int_dt(su.dtype, su.data, s * su.stride[0]) - qlen + i;
        const int64_t hi = std::min(pos + 1, ctx_max);
        const int64_t lo = window > 0 ? std::max<int64_t>(0, pos - window + 1) : 0;
        mla_sink_one(c, r, head, s, lo, hi, ldt(&sinks, head * sinks.stride[0]));
    }
    return RAD_OK;
}

/* mla_attn's per-query table with a rotary part and a sink: tokens [0, length[r]) of row r's own
 * table row, page bt[r][t / block_size], slot t % block_size. */
AVX_KERNEL(mla_attn_sel_sink) {
    if (a->n_t < 7 || !rad_args_have(a, 7)) return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[6];
    const RadTensor &kv = a->t[2], &bt = a->t[3], &len = a->t[4], &sinks = a->t[5];
    const int64_t H = p_int(a, "heads", 0), L = p_int(a, "latent_dim", 0),
                  R = p_int(a, "rope_dim", 0), bs = p_int(a, "block_size", 0);
    const float scale = p_f32(a, "scale", 0);
    heads_view(q, H, L);
    heads_view(qr, H, R);
    heads_view(o, H, L);
    if (H < 1 || L < 1 || R < 1 || bs < 1 || !std::isfinite(scale) || scale <= 0 || q.rank != 3 ||
        q.shape[0] < 1 || q.shape[1] != H || q.shape[2] != L || qr.rank != 3 ||
        qr.shape[0] != q.shape[0] || qr.shape[1] != H || qr.shape[2] != R || kv.rank != 4 ||
        kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < L + R || bt.rank != 2 ||
        bt.shape[0] < q.shape[0] || len.rank != 1 || len.shape[0] < q.shape[0] || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != H || o.shape[2] != L || !sinks_ok(sinks, H))
        return RAD_E_SHAPE;
    const int64_t rows = q.shape[0];
    const MlaSink c{ &q, &qr, &kv, &bt, &o, L, R, bs, scale };
    const int64_t ctx_max = bt.shape[1] * bs;
    HY4_PAR_DYN(rows * H * (L + R) * std::min<int64_t>(ctx_max, 4096))
    for (int64_t job = 0; job < rows * H; ++job) {
        const int64_t r = job / H, head = job % H;
        const int64_t hi = std::clamp(ld_int_dt(len.dtype, len.data, r * len.stride[0]),
                                      (int64_t)0, ctx_max);
        mla_sink_one(c, r, head, r, 0, hi, ldt(&sinks, head * sinks.stride[0]));
    }
    return RAD_OK;
}

/* ================================================================== non-absorbed prefill */

/* kvb [N, H (kd+vd)] and k_rope [N, R] -> k [N, H, kd+R] (the rope part shared by every head) and
 * v [N, H, vd]. A copy; operands of other dtypes are converted. */
AVX_KERNEL(mla_kv_split) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor &kvb = a->t[0], &kr = a->t[1];
    RadTensor k = a->t[2], v = a->t[3];
    const int64_t H = p_int(a, "heads", 0), kd = p_int(a, "k_dim", 0), vd = p_int(a, "v_dim", 0),
                  R = p_int(a, "rope_dim", 0);
    if (H < 1 || kd < 1 || vd < 1 || R < 1) return RAD_E_SHAPE;
    heads_view(k, H, kd + R);
    heads_view(v, H, vd);
    if (kvb.rank != 2 || kvb.shape[0] < 1 || kvb.shape[1] != H * (kd + vd) || kr.rank != 2 ||
        kr.shape[0] != kvb.shape[0] || kr.shape[1] != R || k.rank != 3 ||
        k.shape[0] != kvb.shape[0] || k.shape[1] != H || k.shape[2] != kd + R || v.rank != 3 ||
        v.shape[0] != kvb.shape[0] || v.shape[1] != H || v.shape[2] != vd)
        return RAD_E_SHAPE;
    const int64_t N = kvb.shape[0];
    /* Same dtype and unit strides: the bytes move as they are. */
    const bool bits = kvb.dtype == k.dtype && kr.dtype == k.dtype && v.dtype == k.dtype &&
                      kvb.stride[1] == 1 && kr.stride[1] == 1 && k.stride[2] == 1 &&
                      v.stride[2] == 1 && rad_dtype_bits(k.dtype) % 8 == 0;
    const size_t eb = (size_t)(rad_dtype_bits(k.dtype) / 8);
    AVX_PARALLEL_FOR_IF(N * H * (kd + R + vd) >= AVX_PAR_MIN)
    for (int64_t i = 0; i < N; ++i) {
        const int64_t kr0 = i * kr.stride[0];
        float* tmp = scratch_f32((size_t)(kd + vd + R + 16));
        const float* rope = bits ? nullptr : in_row(&kr, kr0, R, tmp + kd + vd);
        for (int64_t h = 0; h < H; ++h) {
            const int64_t src = i * kvb.stride[0] + h * (kd + vd) * kvb.stride[1];
            const int64_t kdst = i * k.stride[0] + h * k.stride[1];
            const int64_t vdst = i * v.stride[0] + h * v.stride[1];
            if (bits) {
                const uint8_t* s = (const uint8_t*)kvb.data + (size_t)src * eb;
                uint8_t* kd8 = (uint8_t*)k.data + (size_t)kdst * eb;
                uint8_t* vd8 = (uint8_t*)v.data + (size_t)vdst * eb;
                std::memcpy(kd8, s, (size_t)kd * eb);
                std::memcpy(kd8 + (size_t)kd * eb, (const uint8_t*)kr.data + (size_t)kr0 * eb,
                            (size_t)R * eb);
                std::memcpy(vd8, s + (size_t)kd * eb, (size_t)vd * eb);
            } else {
                const float* w = in_row(&kvb, src, kd + vd, tmp);
                out_row(&k, kdst, kd, w);
                out_row(&k, kdst + kd * k.stride[2], R, rope);
                out_row(&v, vdst, vd, w + kd);
            }
        }
    }
    return RAD_OK;
}

/* Dense attention over a key prefix, with a window and a sink. Sequence s has m_s query rows
 * [cu_q[s], cu_q[s+1]) and n_s >= m_s key rows [cu_k[s], cu_k[s+1]); query i sits at key index
 * p = n_s - m_s + i and sees j <= p with p - j < window (window 0: all). */
AVX_KERNEL(attn_kv_sink) {
    if (a->n_t < 7 || !rad_args_have(a, 7)) return RAD_E_INVAL;
    RadTensor q = a->t[0], k = a->t[1], v = a->t[2], o = a->t[6];
    const RadTensor &cuq = a->t[3], &cuk = a->t[4], &sinks = a->t[5];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0),
                  window = p_int(a, "window", 0);
    const float scale = p_f32(a, "scale", 0);
    if (H < 1 || D < 1 || window < 0 || !std::isfinite(scale) || scale <= 0) return RAD_E_SHAPE;
    heads_view(q, H, D);
    heads_view(k, H, D);
    if (v.rank == 2 && v.shape[1] % H == 0) heads_view(v, H, v.shape[1] / H);
    if (o.rank == 2 && o.shape[1] % H == 0) heads_view(o, H, o.shape[1] / H);
    if (q.rank != 3 || k.rank != 3 || v.rank != 3 || o.rank != 3 || q.shape[0] < 1 ||
        k.shape[0] < 1 || q.shape[1] != H || q.shape[2] != D || k.shape[1] != H || k.shape[2] != D ||
        v.shape[0] != k.shape[0] || v.shape[1] != H || o.shape[0] != q.shape[0] ||
        o.shape[1] != H || o.shape[2] != v.shape[2] || cuq.rank != 1 || cuk.rank != 1 ||
        numel(&cuq) != numel(&cuk) || numel(&cuq) < 2 || !sinks_ok(sinks, H))
        return RAD_E_SHAPE;
    const int64_t V = v.shape[2], n_seq = numel(&cuq) - 1, rows = q.shape[0];
    if (cuv(&cuq, 0) != 0 || cuv(&cuq, n_seq) != rows) return RAD_E_SHAPE;
    int64_t longest = 0;
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t m = cuv(&cuq, s + 1) - cuv(&cuq, s);
        const int64_t n = cuv(&cuk, s + 1) - cuv(&cuk, s);
        if (m < 0 || n < m || cuv(&cuk, s) < 0 || cuv(&cuk, s + 1) > k.shape[0]) return RAD_E_SHAPE;
        longest = std::max(longest, n);
    }
    HY4_PAR_DYN(rows * H * (D + V) * std::min<int64_t>(longest, 4096))
    for (int64_t job = 0; job < rows * H; ++job) {
        const int64_t r = job / H, head = job % H;
        const int64_t s = seq_row(&cuq, n_seq, r);
        const int64_t m = cuv(&cuq, s + 1) - cuv(&cuq, s), n = cuv(&cuk, s + 1) - cuv(&cuk, s);
        const int64_t k0 = cuv(&cuk, s), p = n - m + (r - cuv(&cuq, s));
        const int64_t lo = window > 0 ? std::max<int64_t>(0, p - window + 1) : 0;
        float* tmp = scratch_f32((size_t)(2 * D + 2 * V + 16));
        float *qb = tmp, *kb = qb + D, *vb = kb + D, *value = vb + V;
        const float* qv = in_row(&q, r * q.stride[0] + head * q.stride[1], D, qb);
        for (int64_t i = 0; i < V; ++i) value[i] = 0;
        const float sink = ldt(&sinks, head * sinks.stride[0]);
        float mx = sink, den = 0;
        for (int64_t j = lo; j <= p; ++j) {
            const float* key = in_row(&k, (k0 + j) * k.stride[0] + head * k.stride[1], D, kb);
            const float* val = in_row(&v, (k0 + j) * v.stride[0] + head * v.stride[1], V, vb);
            const float sc = dotf(qv, key, D) * scale;
            float w;
            if (sc > mx) {
                const float old = std::exp(mx - sc);
                const vf ov = vf_set1(old);
                int64_t i = 0;
                for (; i + VF_N <= V; i += VF_N) vf_storeu(value + i, vf_mul(vf_loadu(value + i), ov));
                for (; i < V; ++i) value[i] *= old;
                den *= old;
                mx = sc;
                w = 1.0f;
            } else {
                w = std::exp(sc - mx);
            }
            den += w;
            const vf wv = vf_set1(w);
            int64_t i = 0;
            for (; i + VF_N <= V; i += VF_N)
                vf_storeu(value + i, vf_fma(vf_loadu(val + i), wv, vf_loadu(value + i)));
            for (; i < V; ++i) value[i] += w * val[i];
        }
        den += std::exp(sink - mx);
        if (p >= lo) {
            const vf dv = vf_set1(den);
            int64_t i = 0;
            for (; i + VF_N <= V; i += VF_N) vf_storeu(value + i, vf_div(vf_loadu(value + i), dv));
            for (; i < V; ++i) value[i] /= den;
        } else {
            for (int64_t i = 0; i < V; ++i) value[i] = 0;
        }
        out_row(&o, r * o.stride[0] + head * o.stride[1], V, value);
    }
    return RAD_OK;
}

/* ================================================================== gate_mul */

/* y = round(f32(a) * sigmoid(f32(g))): one rounding. */
AVX_KERNEL(gate_mul) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &g = a->t[1], &y = a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n < 1 || y.rank < 1) return RAD_E_SHAPE;
    const int64_t M = rows_of(&y, n);
    if (M < 1 || numel(&x) < M * n || numel(&g) < M * n || y.shape[y.rank - 1] != n)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        float* tmp = scratch_f32((size_t)(3 * n + 16));
        float *xb = tmp, *gb = tmp + n, *yb = tmp + 2 * n;
        const float* xv = in_row(&x, rowoff(&x, r, n), n, xb);
        const float* gv = in_row(&g, rowoff(&g, r, n), n, gb);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(yb + i, vf_mul(vf_loadu(xv + i), vf_sigmoid(vf_loadu(gv + i))));
        for (; i < n; ++i) yb[i] = xv[i] * sigmoid1(gv[i]);
        out_row(&y, rowoff(&y, r, n), n, yb);
    }
    return RAD_OK;
}

/* ================================================================== mla_expand_gate */

/* mla_project_kv's expand into y, then gate_mul over y in place: the pair it stands for, literally
 * (libref's, on libavx's two kernels: a host graph that declares the fusion runs what the pair
 * would). Operands: x, w, g [M, heads * v_dim], y [M, heads, v_dim]. */
AVX_KERNEL(mla_expand_gate) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const int64_t H = p_int(a, "heads", 0), vd = p_int(a, "v_dim", 0);
    if (H < 1 || vd < 1) return RAD_E_SHAPE;
    std::vector<RadParam> pe(a->p, a->p + a->n_p);
    RadParam mode{};
    mode.key = "mode"; mode.kind = RAD_P_STR; mode.sval = "expand";
    pe.push_back(mode);
    RadTensor te[3] = { a->t[0], a->t[1], a->t[3] };
    RadArgs e = *a;
    e.t = te; e.n_t = 3; e.p = pe.data(); e.n_p = (int)pe.size();
    const int rc = AVX_FN(avx_mla_project_kv)(&e, stream);
    if (rc != RAD_OK) return rc;
    /* y as [M, heads * v_dim]: a token's heads are one row */
    RadTensor y = a->t[3];
    if (y.rank == 3) {
        if (y.stride[2] != 1 || y.stride[1] != vd || y.shape[1] != H || y.shape[2] != vd) return RAD_E_STRIDE;
        y.rank = 2; y.shape[1] = H * vd; y.stride[1] = 1;
    }
    RadParam pg[2] = {};
    pg[0].key = "n"; pg[0].kind = RAD_P_INT; pg[0].ival = H * vd;
    pg[1].key = "dtype"; pg[1].kind = RAD_P_STR; pg[1].sval = p_str(a, "dtype", "bf16");
    RadTensor tg[3] = { y, a->t[2], y };
    RadArgs g = *a;
    g.t = tg; g.n_t = 3; g.p = pg; g.n_p = 2;
    return AVX_FN(avx_gate_mul)(&g, stream);
}

/* ================================================================== swiglu_clamped_had_quant_i8g */

/* swiglu_clamped into the bf16 product (`out_bf16`, or a scratch plane), then had_quant_act_i8g over
 * it: the pair itself, on libavx's two kernels (libref's fused op's chain). Operands: gate_up
 * [M, 2n], q [M, n], scale [M, n / group], out_bf16? [M, n]. */
AVX_KERNEL(swiglu_clamped_had_quant_i8g) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    if (n < 1 || a->t[0].rank != 2 || a->t[0].shape[1] != 2 * n) return RAD_E_SHAPE;
    const int64_t M = a->t[0].shape[0];
    std::vector<uint16_t> tmp;
    RadTensor y{};
    if (a->n_t > 3 && a->t[3].data) {
        y = a->t[3];
    } else {
        tmp.resize((size_t)(M * n));
        y.data = tmp.data(); y.dtype = RAD_BF16; y.rank = 2;
        y.shape[0] = M; y.shape[1] = n; y.stride[0] = n; y.stride[1] = 1;
    }
    RadTensor tc[2] = { a->t[0], y };
    RadArgs c = *a;
    c.t = tc; c.n_t = 2;
    const int rc = AVX_FN(avx_swiglu_clamped)(&c, stream);
    if (rc != RAD_OK) return rc;
    RadTensor tq[3] = { y, a->t[1], a->t[2] };
    RadArgs q = *a;
    q.t = tq; q.n_t = 3;
    return AVX_FN(avx_had_quant_act_i8g)(&q, stream);
}


/* mla_expand_gate_had_quant_i8g: mla_expand_gate into the bf16 product (`out_bf16`, or a scratch plane), then
 * had_quant_act_i8g over it as [M, heads * v_dim]: the chain itself. Operands: x, w, g [M, heads *
 * v_dim], q [M, heads * v_dim], scale [M, heads * v_dim / group], out_bf16? */
AVX_KERNEL(mla_expand_gate_had_quant_i8g) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const int64_t H = p_int(a, "heads", 0), vd = p_int(a, "v_dim", 0);
    if (H < 1 || vd < 1) return RAD_E_SHAPE;
    const int64_t M = a->t[0].shape[0], n = H * vd;
    std::vector<uint16_t> tmp;
    int64_t pitch = n;
    uint16_t* yd = nullptr;
    if (a->n_t > 5 && a->t[5].data) {
        const RadTensor& o = a->t[5];
        if (o.shape[o.rank - 1] != vd && (o.rank != 2 || o.shape[1] != n)) return RAD_E_SHAPE;
        if (o.stride[o.rank - 1] != 1) return RAD_E_STRIDE;
        yd = (uint16_t*)o.data;
        pitch = o.rank == 3 ? o.stride[0] : o.stride[0];
    } else {
        tmp.resize((size_t)(M * n));
        yd = tmp.data();
    }
    RadTensor y3{};
    y3.data = yd; y3.dtype = RAD_BF16; y3.rank = 3;
    y3.shape[0] = M; y3.shape[1] = H; y3.shape[2] = vd;
    y3.stride[0] = pitch; y3.stride[1] = vd; y3.stride[2] = 1;
    RadTensor te[4] = { a->t[0], a->t[1], a->t[2], y3 };
    RadArgs e = *a;
    e.t = te; e.n_t = 4;
    int rc = AVX_FN(avx_mla_expand_gate)(&e, stream);
    if (rc != RAD_OK) return rc;
    RadTensor y2 = y3;
    y2.rank = 2; y2.shape[1] = n; y2.stride[1] = 1;
    RadTensor tq[3] = { y2, a->t[3], a->t[4] };
    RadParam pq[4] = {};
    pq[0].key = "M"; pq[0].kind = RAD_P_INT; pq[0].ival = M;
    pq[1].key = "n"; pq[1].kind = RAD_P_INT; pq[1].ival = n;
    pq[2].key = "group"; pq[2].kind = RAD_P_INT; pq[2].ival = p_int(a, "group", 128);
    pq[3].key = "dtype"; pq[3].kind = RAD_P_STR; pq[3].sval = "bf16";
    RadArgs q = *a;
    q.t = tq; q.n_t = 3; q.p = pq; q.n_p = 4;
    return AVX_FN(avx_had_quant_act_i8g)(&q, stream);
}

/* ================================================================== the compressed indexer */

namespace {

/* Rotary pair j's position component for imrope_gptj with sections (t, h, w). */
int imrope_component(int64_t j, int64_t sec_h, int64_t sec_w) {
    if (j % 3 == 1 && j < 3 * sec_h) return 1;
    if (j % 3 == 2 && j < 3 * sec_w) return 2;
    return 0;
}

}  // namespace

/* Pools each closed group of `ratio` token keys into one rotated key: gated softmax over the slots
 * per channel, an rms norm with a gain, the rotary on the last `rotary_dim` channels at the group's
 * first position; rounded to k's dtype after the norm and after the rotation, as HF does. */
namespace {
/* `fp8` 0: idx_pool_gated. `fp8` 1: idx_pool_gated_fp8, the key kept in f32 (nothing rounded), then
 * the ascending-stride Walsh-Hadamard transform times n^-0.5 and E4M3 codes at a power-of-two
 * scale, into dsa_pool's packed cache page and/or q/scale (libref's ref_idx_pool_gated_fp8). */
int pool_gated(const RadArgs* a, bool fp8, bool i8 = false) {
    if (a->n_t < 6 || !rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor &k = a->t[0], &wv = a->t[1], &wg = a->t[2], &ape = a->t[3], &gain = a->t[4],
                    &pos = a->t[5];
    const RadTensor* page = a->n_t > 6 ? rad_arg_in(a, 6) : nullptr;
    const RadTensor* cache = a->n_t > 7 ? rad_arg_in(a, 7) : nullptr;
    const RadTensor* out = !fp8 && a->n_t > 8 ? rad_arg_in(a, 8) : nullptr;
    const RadTensor* q = fp8 && a->n_t > 8 ? rad_arg_in(a, 8) : nullptr;
    const RadTensor* qs = fp8 && a->n_t > 9 ? rad_arg_in(a, 9) : nullptr;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), rd = p_int(a, "rotary_dim", 0);
    const float eps = p_f32(a, "eps", 0);
    const double theta = (double)p_f32(a, "theta", 0);
    const char* mode = p_str(a, "mode", "");
    const bool gptj = !std::strcmp(mode, "gptj"), imrope = !std::strcmp(mode, "imrope_gptj");
    if (n < 1 || ratio < 1 || ratio > 16 || rd < 2 || rd > n || (rd & 1) || !(eps > 0) ||
        !std::isfinite(eps) || !(theta > 1.0) || (!gptj && !imrope))
        return RAD_E_SHAPE;
    int64_t sec_h = 0, sec_w = 0;
    if (imrope) {
        const char* s = p_str(a, "sections", nullptr);
        long t = 0, h = 0, w = 0;
        if (!s || std::sscanf(s, "%ld %ld %ld", &t, &h, &w) != 3 || t < 0 || h < 0 || w < 0 ||
            t + h + w != rd / 2)
            return RAD_E_SHAPE;
        sec_h = h;
        sec_w = w;
    }
    if (!page != !cache) return RAD_E_INVAL;
    if (!fp8 && !cache && !out) return RAD_E_INVAL;
    if (fp8 && (!q != !qs || (!cache && !q))) return RAD_E_INVAL;
    if (k.rank != 2 || k.shape[0] < ratio || k.shape[0] % ratio || k.shape[1] != n)
        return RAD_E_SHAPE;
    const int64_t W = k.shape[0] / ratio;
    if (!matrix(wv, n, n) || !matrix(wg, n, n) || !matrix(ape, ratio, n) || gain.rank != 1 ||
        numel(&gain) != n || pos.rank != (imrope ? 2 : 1) ||
        (imrope ? (pos.shape[1] != 3 || pos.shape[0] < W) : pos.shape[0] < W))
        return RAD_E_SHAPE;
    if (page && (page->rank != 1 || page->shape[0] < W)) return RAD_E_SHAPE;
    if (!fp8 && cache && (cache->rank != 4 || cache->shape[0] < 1 || cache->shape[1] != 1 ||
                  cache->shape[2] * cache->shape[3] != n || cache->stride[3] != 1 ||
                  cache->stride[2] != cache->shape[3] || cache->stride[0] < n))
        return RAD_E_SHAPE;
    if (out && !matrix(*out, W, n)) return RAD_E_SHAPE;
    if (fp8) {
        if (k.dtype != RAD_BF16 && k.dtype != RAD_F32) return RAD_E_DTYPE;
        if (n > 256 || (n & (n - 1))) return RAD_E_SHAPE;
        if (cache) {
            if (page->dtype != RAD_I32 || cache->dtype != RAD_U8) return RAD_E_DTYPE;
            if (n != 128 || cache->rank != 4 || cache->shape[0] < 1 || cache->shape[1] != 1 ||
                cache->shape[2] != 4 || cache->shape[3] != 34 || cache->stride[3] != 1 ||
                cache->stride[2] != 34 || cache->stride[0] < 136 || cache->stride[0] % 4 ||
                ((uintptr_t)cache->data & 3))
                return RAD_E_SHAPE;
        }
        if (q) {
            if (q->dtype != (i8 ? RAD_I8 : RAD_F8E4M3) || qs->dtype != RAD_F32) return RAD_E_DTYPE;
            if (!matrix(*q, W, n) || !matrix(*qs, W, 1)) return RAD_E_SHAPE;
        }
    }
    /* Two groups naming one page would make the result depend on the thread order. */
    if (page) {
        std::vector<int64_t> seen;
        for (int64_t w = 0; w < W; ++w) {
            const int64_t pw = ldi(page, offlin(page, w));
            if (pw < 0 || pw >= cache->shape[0]) continue;
            seen.push_back(pw);
        }
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) return RAD_E_INVAL;
    }
    /* The rotary's cos and sin for the rotated channels at every group, from the pair's inverse
     * frequency in double as libref does. */
    std::vector<float> inv_freq((size_t)(rd / 2));
    for (int64_t i = 0; i < rd / 2; ++i)
        inv_freq[(size_t)i] = (float)(1.0 / std::pow(theta, (double)(2 * i) / (double)rd));
    const float* ifq = inv_freq.data();

    HY4_PAR_DYN(W * 2 * n * n * ratio)
    for (int64_t w = 0; w < W; ++w) {
        /* x_s, val_s and gat_s rows, y, z. */
        float* tmp = scratch_f32((size_t)(ratio * n * 3 + 2 * n + 64));
        float *xs = tmp, *val = xs + ratio * n, *gat = val + ratio * n, *y = gat + ratio * n,
              *z = y + n;
        for (int64_t s = 0; s < ratio; ++s) {
            const float* x = in_row(&k, rowoff(&k, w * ratio + s, n), n, xs + s * n);
            if (x != xs + s * n) std::memcpy(xs + s * n, x, (size_t)n * sizeof(float));
        }
        float* wrow = scratch_f32_b((size_t)(2 * n + 16));
        for (int64_t c = 0; c < n; ++c) {
            const float* rv = in_row(&wv, rowoff(&wv, c, n), n, wrow);
            const float* rg = in_row(&wg, rowoff(&wg, c, n), n, wrow + n);
            for (int64_t s = 0; s < ratio; ++s) {
                val[s * n + c] = dotf(rv, xs + s * n, n);
                gat[s * n + c] = dotf(rg, xs + s * n, n) + ldt(&ape, s * ape.stride[0] + c * ape.stride[1]);
            }
        }
        for (int64_t c = 0; c < n; ++c) {
            float mx = -INFINITY;
            for (int64_t s = 0; s < ratio; ++s) mx = std::max(mx, gat[s * n + c]);
            float e[16], den = 0;
            for (int64_t s = 0; s < ratio; ++s) {
                e[s] = sc_exp(gat[s * n + c] - mx);
                den += e[s];
            }
            float acc = 0;
            for (int64_t s = 0; s < ratio; ++s) acc += val[s * n + c] * (e[s] / den);
            y[c] = acc;
        }
        double sq = 0;
        for (int64_t c = 0; c < n; ++c) sq += (double)y[c] * (double)y[c];
        const float inv = 1.0f / std::sqrt((float)(sq / (double)n) + eps);
        auto rnd = [&](float v) { return fp8 ? v : narrow1(k.dtype, v); };
        for (int64_t c = 0; c < n; ++c) z[c] = rnd(ldt(&gain, c * gain.stride[0]) * (y[c] * inv));
        for (int64_t i = 0; i < rd / 2; ++i) {
            const int comp = imrope ? imrope_component(i, sec_h, sec_w) : 0;
            const int64_t pv = imrope ? ld_int_dt(pos.dtype, pos.data, w * pos.stride[0] + comp * pos.stride[1])
                                      : ld_int_dt(pos.dtype, pos.data, w * pos.stride[0]);
            const float ang = (float)pv * ifq[i];
            const float cs = sc_cos(ang), sn = sc_sin(ang);
            const int64_t o0 = n - rd + 2 * i;
            const float x0 = z[o0], x1 = z[o0 + 1];
            z[o0] = rnd(x0 * cs - x1 * sn);
            z[o0 + 1] = rnd(x1 * cs + x0 * sn);
        }
        if (fp8) {
            for (int64_t h = 1; h < n; h *= 2)
                for (int64_t i = 0; i < n; i += 2 * h)
                    for (int64_t j = 0; j < h; ++j) {
                        const float lo = z[i + j], hi = z[i + j + h];
                        z[i + j] = lo + hi;
                        z[i + j + h] = lo - hi;
                    }
            const float norm = 1.0f / std::sqrt((float)n);
            float am = 1e-4f;
            for (int64_t d = 0; d < n; ++d) {
                z[d] = z[d] * norm;
                am = std::max(am, std::fabs(z[d]));
            }
            float scale;
            uint8_t codes[256];
            if (i8) {
                /* idx_pool_gated_i8: symmetric int8 at an f32 scale amax / 127 (libref). */
                scale = am * (1.0f / 127.0f);
                const float inv = 1.0f / scale;
                for (int64_t d = 0; d < n; ++d)
                    codes[d] = (uint8_t)(int8_t)std::nearbyint(
                        std::clamp(z[d] * inv, -127.0f, 127.0f));
            } else {
                int exponent = 0;
                const float fraction = std::frexp(am * (1.0f / 448.0f), &exponent);
                scale = std::ldexp(1.0f, exponent - (fraction == 0.5f));
                for (int64_t d = 0; d < n; ++d)
                    codes[d] = rad_f32_to_fp8e4m3(std::clamp(z[d] / scale, -448.0f, 448.0f));
            }
            if (q) {
                for (int64_t d = 0; d < n; ++d)
                    ((uint8_t*)q->data)[rowoff(q, w, n) + d * laststride(q)] = codes[d];
                stt(qs, rowoff(qs, w, 1), scale);
            }
            if (cache) {
                const int64_t pg = ldi(page, offlin(page, w));
                if (pg >= 0 && pg < cache->shape[0]) {
                    auto* dst = (uint8_t*)cache->data + pg * cache->stride[0];
                    std::memcpy(dst, codes, 128);
                    std::memcpy(dst + 128, &scale, sizeof(scale));
                }
            }
            continue;
        }
        if (out) out_row(out, rowoff(out, w, n), n, z);
        if (cache) {
            const int64_t pg = ldi(page, offlin(page, w));
            if (pg >= 0 && pg < cache->shape[0]) {
                for (int64_t c = 0; c < n; ++c) stt(cache, pg * cache->stride[0] + c, z[c]);
            }
        }
    }
    return RAD_OK;
}
}  // namespace

AVX_KERNEL(idx_pool_gated) { return pool_gated(a, false); }
AVX_KERNEL(idx_pool_gated_fp8) { return pool_gated(a, true); }
AVX_KERNEL(idx_pool_gated_i8) { return pool_gated(a, true, true); }

/* score[m][b] = sum_h w_h relu(q[m][h] . key_b), w_h = gate[m][h] (heads n)^-0.5, the heads summed
 * as two halves in ascending order and the halves added. Blocks past nc[seq], with a page < 0 or
 * out of range, or past the table: -inf. */
AVX_KERNEL(idx_score) {
    if (a->n_t < 7 || !rad_args_have(a, 7)) return RAD_E_INVAL;
    RadTensor q = a->t[0];
    const RadTensor &gate = a->t[1], &key = a->t[2], &bt = a->t[3], &cu = a->t[4], &nc = a->t[5],
                    &score = a->t[6];
    const int64_t n = p_int(a, "n", 0), H = p_int(a, "heads", 0);
    if (n < 1 || H < 1) return RAD_E_SHAPE;
    heads_view(q, H, n);
    if (q.rank != 3 || q.shape[0] < 1 || q.shape[1] != H || q.shape[2] != n || gate.rank != 2 ||
        gate.shape[0] != q.shape[0] || gate.shape[1] < H || bt.rank != 2 || cu.rank != 1 ||
        cu.shape[0] < 2 || score.rank != 2 || score.shape[0] != q.shape[0] || score.shape[1] < 1)
        return RAD_E_SHAPE;
    const bool paged = key.rank == 4;
    if (paged ? (key.shape[1] != 1 || key.shape[2] * key.shape[3] != n || key.stride[3] != 1 ||
                 key.stride[2] != key.shape[3] || key.stride[0] < n)
              : (key.rank != 2 || key.shape[1] != n))
        return RAD_E_SHAPE;
    const int64_t M = q.shape[0], n_seq = cu.shape[0] - 1, pages = key.shape[0];
    if (bt.shape[0] < n_seq || numel(&nc) < n_seq || cuv(&cu, 0) != 0 || cuv(&cu, n_seq) != M)
        return RAD_E_SHAPE;
    const float wscale = (float)(std::pow((double)H, -0.5) * std::pow((double)n, -0.5));
    const int64_t cols = std::min(score.shape[1], bt.shape[1]);
    const int64_t width = score.shape[1], half = H / 2;
    constexpr int64_t kChunk = 64;
    const int64_t nchunk = (width + kChunk - 1) / kChunk, units = M * nchunk;
    HY4_PAR_DYN(M * cols * H * n)
    for (int64_t u = 0; u < units; ++u) {
        const int64_t m = u / nchunk, b0 = (u % nchunk) * kChunk, b1 = std::min(b0 + kChunk, width);
        const int64_t s = seq_row(&cu, n_seq, m);
        const int64_t nlive = std::min(ldi(&nc, offlin(&nc, s)), cols);
        float* tmp = scratch_f32((size_t)(H * n + n + kChunk + H + 16));
        float *qf = tmp, *kb = qf + H * n, *outv = kb + n, *wh = outv + kChunk;
        for (int64_t h = 0; h < H; ++h) {
            const float* qv = in_row(&q, m * q.stride[0] + h * q.stride[1], n, qf + h * n);
            if (qv != qf + h * n) std::memcpy(qf + h * n, qv, (size_t)n * sizeof(float));
            wh[h] = ldt(&gate, m * gate.stride[0] + h * gate.stride[1]) * wscale;
        }
        for (int64_t b = b0; b < b1; ++b) {
            float res = -INFINITY;
            const int64_t page = b < nlive ? ld_int_dt(bt.dtype, bt.data, s * bt.stride[0] + b * bt.stride[1]) : -1;
            if (b < cols && page >= 0 && page < pages) {
                const float* kv = in_row(&key, page * key.stride[0], n, kb);
                float lo = 0, hi = 0;
                for (int64_t h = 0; h < H; ++h) {
                    const float d = dotf(qf + h * n, kv, n);
                    const float term = wh[h] * std::max(d, 0.0f);
                    (h < half ? lo : hi) += term;
                }
                res = lo + hi;
            }
            outv[b - b0] = res;
        }
        out_row(&score, m * score.stride[0] + b0 * score.stride[1], b1 - b0, outv);
    }
    return RAD_OK;
}

/* ================================================================== Engram */

/* The hashed n-gram ids of one step: libref/ref_hy4.cpp's ref_engram_ids, the same integer
 * arithmetic. Token k places back is the sequence's own token, or pad before the document start;
 * ids at or past the token map read as pad. */
AVX_KERNEL(engram_ids) {
    if (a->n_t < 12 || !rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 2) ||
        !rad_arg_in(a, 3) || !rad_arg_in(a, 4) || !rad_arg_in(a, 5) || !rad_arg_in(a, 6) ||
        !rad_arg_in(a, 7) || !rad_arg_in(a, 10) || !rad_arg_in(a, 11))
        return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &st = a->t[1], &cu = a->t[2], &pos = a->t[3], &map = a->t[4],
                    &mult = a->t[5], &vsz = a->t[6], &off = a->t[7];
    const RadTensor* nacc = rad_arg_in(a, 8);
    const RadTensor* sidx = rad_arg_in(a, 9);
    const RadTensor &ids = a->t[10], &mask = a->t[11];
    /* The following prompt's ids (a read-ahead hint): `tok` holds A tokens of the last sequence
     * past the step's rows. */
    const RadTensor* ahead = a->n_t > 12 ? rad_arg_in(a, 12) : nullptr;
    const int64_t heads = p_int(a, "heads", 0), ngram = p_int(a, "ngram", 0),
                  pad = p_int(a, "pad", -1), msize = p_int(a, "map_size", 0),
                  image = p_int(a, "image", -1), video = p_int(a, "video", -1);
    if (heads < 1 || ngram < 2 || ngram > 8 || heads % (ngram - 1) || msize < 1) return RAD_E_SHAPE;
    if (pad < 0 || pad >= msize) return RAD_E_INVAL;
    const int64_t per = heads / (ngram - 1), ctx = ngram - 1;
    if (tok.rank < 1 || numel(&tok) < 1 || numel(&map) < msize || numel(&mult) < ngram ||
        numel(&vsz) < heads || numel(&off) < heads || cu.rank != 1 || cu.shape[0] < 2)
        return RAD_E_SHAPE;
    const int64_t A = ahead ? numel(ahead) / heads : 0;
    const int64_t M = numel(&tok) - A, n_seq = cu.shape[0] - 1;
    if (M < 1 || numel(&pos) < M || ids.rank != 2 || ids.shape[0] < M || ids.shape[1] != heads ||
        numel(&mask) < M || st.rank < 2 || st.shape[st.rank - 1] < ctx ||
        (ahead && (ahead->rank != 2 || ahead->shape[1] != heads || cuv(&cu, n_seq) != M)))
        return RAD_E_SHAPE;
    std::vector<int64_t> vtab((size_t)heads), otab((size_t)heads), mtab((size_t)ngram);
    for (int64_t h = 0; h < heads; ++h) {
        vtab[(size_t)h] = ldi(&vsz, offlin(&vsz, h));
        otab[(size_t)h] = ldi(&off, offlin(&off, h));
        if (vtab[(size_t)h] <= 0 || otab[(size_t)h] < 0) return RAD_E_SHAPE;
    }
    for (int64_t k = 0; k < ngram; ++k) mtab[(size_t)k] = ldi(&mult, offlin(&mult, k));
    /* The CONV group's state: [slots, W], or [slots, 1, W] as the engine hands it. */
    const int64_t W = st.shape[st.rank - 1];
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t o = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
        const int64_t b0 = cuv(&cu, s), b1 = cuv(&cu, s + 1);
        if (o < 0) return RAD_E_INVAL;
        if (o + ctx > W || b0 < 0 || b1 < b0 || b1 > M) return RAD_E_SHAPE;
        if (nacc && b1 > b0 && ctx - 1 + (b1 - b0) > W) return RAD_E_SHAPE;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
        if (slot >= st.shape[0]) return RAD_E_SHAPE;
    }
    const int64_t map_pad = ldi(&map, offlin(&map, pad));
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t bos = cuv(&cu, s), T = cuv(&cu, s + 1) - bos;
        if (T <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        const int64_t o = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
        const int64_t sb = rowoff(&st, slot, W);
        int64_t prev[8];
        for (int64_t i = 0; i < ctx; ++i) prev[i] = ld_int_dt(st.dtype, st.data, sb + o + i);
        auto at = [&](int64_t t, int64_t k) -> int64_t {
            const int64_t p = ctx + t - k;
            return p < ctx ? prev[p] : ldi(&tok, offlin(&tok, bos + p - ctx));
        };
        /* The last sequence runs on into the following prompt: rows T .. T + A - 1 of it, at the
         * positions after its last, written to `ahead`. */
        const int64_t TA = T + (s + 1 == n_seq ? A : 0);
        AVX_PARALLEL_FOR_IF(TA * heads >= 4096)
        for (int64_t t = 0; t < TA; ++t) {
            const int64_t m = bos + t;
            const int64_t pm = t < T ? ldi(&pos, offlin(&pos, m))
                                     : ldi(&pos, offlin(&pos, bos + T - 1)) + 1 + (t - T);
            const RadTensor& out = t < T ? ids : *ahead;
            const int64_t mo = t < T ? m : t - T;
            int64_t mixed[8];
            for (int64_t k = 0; k < ngram; ++k) {
                int64_t c = map_pad;
                if (pm >= k) {
                    const int64_t tk = at(t, k);
                    if (tk >= 0 && tk < msize) c = ldi(&map, offlin(&map, tk));
                }
                mixed[k] = wrap_mul(c, mtab[(size_t)k]);
            }
            int64_t mix = mixed[0];
            for (int64_t g = 1; g < ngram; ++g) {
                mix ^= mixed[g];
                for (int64_t j = 0; j < per; ++j) {
                    const int64_t h = (g - 1) * per + j;
                    st_int_dt(out.dtype, out.data, mo * out.stride[0] + h * out.stride[1],
                              floor_mod(mix, vtab[(size_t)h]) + otab[(size_t)h]);
                }
            }
            if (t >= T) continue;
            const int64_t t0 = ldi(&tok, offlin(&tok, m));
            st_int_dt(mask.dtype, mask.data, offlin(&mask, m),
                      t0 != pad && t0 != image && t0 != video ? 1 : 0);
        }
        /* The window after the step, by ple_conv's rule: a decode step (num_accepted present)
         * shifts the old ids down by one and appends every id it ran; otherwise every id commits
         * and the last ctx land at offset zero. */
        if (nacc) {
            for (int64_t i = 0; i + 1 < ctx; ++i) st_int_dt(st.dtype, st.data, sb + i, prev[i + 1]);
            for (int64_t t = 0; t < T; ++t)
                st_int_dt(st.dtype, st.data, sb + ctx - 1 + t, ldi(&tok, offlin(&tok, bos + t)));
        } else {
            for (int64_t i = 0; i < ctx; ++i) {
                const int64_t src = T - ctx + i;
                st_int_dt(st.dtype, st.data, sb + i,
                          src >= 0 ? ldi(&tok, offlin(&tok, bos + src)) : prev[src + ctx]);
            }
        }
    }
    return RAD_OK;
}

/* One rms a head: y[i] = round(gain[i] * x[i] * rsqrt(mean_head(x^2) + eps)). */
AVX_KERNEL(engram_norm) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &e = a->t[0], &gain = a->t[1], &y = a->t[2];
    const int64_t H = p_int(a, "heads", 0), n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 0);
    if (H < 1 || n < 1 || !std::isfinite(eps) || eps < 0 || e.rank != 2 || e.shape[0] < 1)
        return RAD_E_SHAPE;
    const int64_t M = e.shape[0], W = H * n;
    if (!matrix(e, M, W) || !matrix(y, M, W) || gain.rank != 1 || numel(&gain) != W)
        return RAD_E_SHAPE;
    /* The gain widened once; a second scratch_raw call could move this block under the workers. */
    float* g = (float*)scratch_raw((size_t)W * sizeof(float) + 64);
    {
        float* tmp = scratch_f32((size_t)W + 16);
        const float* gv = in_row(&gain, 0, W, tmp);
        std::memcpy(g, gv, (size_t)W * sizeof(float));
    }
    AVX_PARALLEL_FOR_IF(M * W >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* tmp = scratch_f32((size_t)(2 * W + 16));
        float *xb = tmp, *yb = tmp + W;
        const float* x = in_row(&e, m * e.stride[0], W, xb);
        for (int64_t h = 0; h < H; ++h) {
            const float ms = dotf(x + h * n, x + h * n, n) / (float)n;
            const float inv = 1.0f / std::sqrt(ms + eps);
            const vf iv = vf_set1(inv);
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(yb + h * n + i,
                          vf_mul(vf_mul(vf_loadu(g + h * n + i), vf_loadu(x + h * n + i)), iv));
            for (; i < n; ++i) yb[h * n + i] = g[h * n + i] * x[h * n + i] * inv;
        }
        out_row(&y, m * y.stride[0], W, yb);
    }
    return RAD_OK;
}

/* The gate of HYV4FlashEngramLayer.forward and its add into the streams. The sums of squares and
 * the dot are f64 and ascending, as libref's (the dot feeds a signed square root). Rows with mask 0
 * are left untouched. */
AVX_KERNEL(engram_gate) {
    if (a->n_t < 8 || !rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor &key = a->t[0], &val = a->t[1], &mask = a->t[2], &wk = a->t[3], &wq = a->t[4],
                    &wv = a->t[5], &wb = a->t[6], &h = a->t[7];
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", 0);
    if (n < 1 || hc < 1 || !std::isfinite(eps) || eps < 0 || h.rank != 2 || h.shape[0] < 1)
        return RAD_E_SHAPE;
    const int64_t M = h.shape[0], W = hc * n;
    if (!matrix(h, M, W) || !matrix(key, M, W) || !matrix(val, M, n) || numel(&mask) < M ||
        numel(&wk) != W || numel(&wq) != W || numel(&wv) != n || numel(&wb) != W)
        return RAD_E_SHAPE;
    const float inv_sqrt_n = (float)std::pow((double)n, -0.5);
    /* The four gains widened once (see engram_norm). */
    float* gains = (float*)scratch_raw((size_t)(3 * W + n) * sizeof(float) + 64);
    float *gk = gains, *gq = gains + W, *gb = gains + 2 * W, *gv = gains + 3 * W;
    {
        float* tmp = scratch_f32((size_t)W + 16);
        auto widen = [&](const RadTensor& t, int64_t cnt, float* dst) {
            const float* s = in_row(&t, 0, cnt, tmp);
            std::memcpy(dst, s, (size_t)cnt * sizeof(float));
        };
        widen(wk, W, gk);
        widen(wq, W, gq);
        widen(wb, W, gb);
        widen(wv, n, gv);
    }
    AVX_PARALLEL_FOR_IF(M * W * 4 >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        if (!ldi(&mask, offlin(&mask, m))) continue;
        float* tmp = scratch_f32((size_t)(3 * W + 3 * n + 64));
        float *kbuf = tmp, *hbuf = kbuf + W, *upd = hbuf + W, *main_ = upd + W, *vbuf = main_ + n;
        const float* kk = in_row(&key, m * key.stride[0], W, kbuf);
        const float* hh = in_row(&h, m * h.stride[0], W, hbuf);
        if (hh != hbuf) std::memcpy(hbuf, hh, (size_t)W * sizeof(float));
        const float* vv = in_row(&val, m * val.stride[0], n, vbuf);
        {
            double sq = 0;
            for (int64_t i = 0; i < n; ++i) sq += (double)vv[i] * (double)vv[i];
            const float inv = 1.0f / std::sqrt((float)(sq / (double)n) + eps);
            for (int64_t i = 0; i < n; ++i) main_[i] = gv[i] * vv[i] * inv;
        }
        for (int64_t c = 0; c < hc; ++c) {
            const float* kc = kk + c * n;
            const float* hcs = hbuf + c * n;
            double sk = 0, sh = 0;
            for (int64_t i = 0; i < n; ++i) {
                sk += (double)kc[i] * (double)kc[i];
                sh += (double)hcs[i] * (double)hcs[i];
            }
            const float ik = 1.0f / std::sqrt((float)(sk / (double)n) + eps);
            const float iq = 1.0f / std::sqrt((float)(sh / (double)n) + eps);
            double dot = 0;
            for (int64_t i = 0; i < n; ++i) {
                const float kv = gk[c * n + i] * kc[i] * ik;
                const float qv = gq[c * n + i] * hcs[i] * iq;
                dot += (double)kv * (double)qv;
            }
            const float d = (float)dot * inv_sqrt_n;
            const float g = sigmoid1(sign_of(d) * std::sqrt(std::max(std::fabs(d), 1e-6f)));
            const vf gvv = vf_set1(g);
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N) {
                const vf mv = vf_loadu(main_ + i);
                vf_storeu(upd + c * n + i,
                          vf_add(vf_mul(gvv, mv), vf_mul(vf_loadu(gb + c * n + i), mv)));
            }
            for (; i < n; ++i) upd[c * n + i] = g * main_[i] + gb[c * n + i] * main_[i];
        }
        /* The inner bf16(...) of the residual, then the add to the stream. */
        round_through(h.dtype, upd, W, (uint16_t*)kbuf);
        for (int64_t i = 0; i < W; ++i) hbuf[i] += upd[i];
        out_row(&h, m * h.stride[0], W, hbuf);
    }
    return RAD_OK;
}

/* ================================================================== DFly2 */

/* dfly_sel: libref/ref_dfly.cpp's draft table, the same rule and the same failure rows. A row gets
 * the source row's recalled pages (sel[src][0, nsel-1), or bt[0, g0) for a densely attended source),
 * then bt[g0] .. bt[pos/ratio], -1 to `blocks`; len and the optional entry count. Integer only:
 * two copies and a fill a row. */
AVX_KERNEL(dfly_sel) {
    if (!a || a->n_t < 8) return RAD_E_INVAL;
    for (int i = 0; i < 8; ++i)
        if (!rad_arg_in(a, i)) return RAD_E_INVAL;
    const RadTensor &sel = a->t[0], &nsel = a->t[1], &src = a->t[2], &bt = a->t[3], &pos = a->t[4],
                    &cu = a->t[5], &out = a->t[6], &len = a->t[7];
    const RadTensor* cnt = rad_arg_in(a, 8);
    const int64_t ratio = p_int(a, "ratio", 0), blocks = p_int(a, "blocks", 0),
                  blocks_in = p_int(a, "blocks_in", 0);
    if (ratio <= 0 || blocks <= 0 || blocks_in <= 0) return RAD_E_SHAPE;
    if (sel.rank != 2 || bt.rank != 2 || out.rank != 2) return RAD_E_SHAPE;
    const int64_t M = out.shape[0], R = sel.shape[0], pitch = bt.shape[1];
    const int64_t n_seq = numel(&cu) - 1;
    if (M <= 0 || n_seq <= 0) return RAD_E_SHAPE;
    if (out.shape[1] < blocks || sel.shape[1] < blocks_in) return RAD_E_SHAPE;
    if (numel(&pos) < M || numel(&len) < M || numel(&src) < n_seq || bt.shape[0] < n_seq ||
        numel(&nsel) < R || (cnt && numel(cnt) < M))
        return RAD_E_SHAPE;
    const int64_t o_rs = out.stride[0], o_cs = out.stride[1];
    const int64_t s_rs = sel.stride[0], s_cs = sel.stride[1];
    const int64_t b_rs = bt.stride[0], b_cs = bt.stride[1];
    auto put = [&](const RadTensor& t, int64_t off, int64_t v) { st_int_dt(t.dtype, t.data, off, v); };
    /* Rows no sequence holds stay empty. */
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t c = 0; c < blocks; ++c) put(out, r * o_rs + c * o_cs, -1);
        put(len, offlin(&len, r), 0);
        if (cnt) put(*cnt, offlin(cnt, r), 0);
    }
    bool overflow = false;
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t r0 = std::max<int64_t>(0, cuv(&cu, s)), r1 = std::min(M, cuv(&cu, s + 1));
        if (r0 >= r1) continue;
        const int64_t anc = ldi(&pos, offlin(&pos, r0));
        const int64_t g0 = anc > 0 ? (anc - 1) / ratio : 0;
        const int64_t k = ldi(&src, offlin(&src, s));
        bool ok = k >= 0 && k < R && anc >= 0;
        const int64_t n = ok ? ldi(&nsel, offlin(&nsel, k)) : 0;
        const int64_t nrec = n > 0 ? n - 1 : g0;
        if (ok && n > 0 && nrec > blocks_in) {
            ok = false;
            overflow = true;
        }
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t p = ldi(&pos, offlin(&pos, r));
            const int64_t e = p / ratio - g0;
            bool row = ok && p >= anc && e >= 0;
            if (row && (nrec + e + 1 > blocks || g0 + e >= pitch || (n <= 0 && nrec > pitch))) {
                row = false;
                overflow = true;
            }
            if (!row) continue;
            int64_t c = 0;
            if (n > 0)
                for (; c < nrec; ++c) put(out, r * o_rs + c * o_cs, ldi(&sel, k * s_rs + c * s_cs));
            else
                for (; c < nrec; ++c) put(out, r * o_rs + c * o_cs, ldi(&bt, s * b_rs + c * b_cs));
            for (int64_t j = 0; j <= e; ++j, ++c)
                put(out, r * o_rs + c * o_cs, ldi(&bt, s * b_rs + (g0 + j) * b_cs));
            put(len, offlin(&len, r), nrec * ratio + (p - g0 * ratio) + 1);
            if (cnt) put(*cnt, offlin(cnt, r), nrec + e + 1);
        }
    }
#ifndef NDEBUG
    if (overflow) return RAD_E_SHAPE;
#else
    (void)overflow;
#endif
    return RAD_OK;
}
