/* tests/r4d_emu: moe_gemm_q at nvfp4a8i8 (r4d_moe_nvfp4.hip) on the host SIMT emulator, against an
 * f64 oracle of the value e2m1 * e4m3 * level times an int8 activation under its f32 scale.
 *
 * The scale operand is the blob the layout hook produces (r4d_layout.cpp, nvfp4): per expert N rows
 * of K / 16 E4M3 bytes, then one row whose first four bytes are the expert's f32 level. The codes
 * are the e2m1 plane as stored. Stacked and pointer-table operands both run, with a token-order and
 * a sorted-order gather, a dropped slot, and rows past the last expert's offset. A tensor-parallel
 * rank's split gate_up runs too: the odd class narrower (N_odd, K_odd) and the output two parts of
 * the wider class's width, the narrower class's columns past its part width zero. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kE2m1Mag2[8] = { 0, 1, 2, 3, 4, 6, 8, 12 };   /* e2m1 magnitudes, doubled */

float e4m3_value(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? (float)m * std::ldexp(1.0f, -9)
                           : (1.0f + (float)m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}

RadTensor tensor(void* data, uint32_t dtype, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[(size_t)i];
        t.stride[i] = st;
        st *= shape[(size_t)i];
    }
    return t;
}

RadParam pint(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam pstr(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

struct Case {
    int N, K, E, top_k, rows;   /* output width, K, experts, slots per token, sorted rows */
    bool pointer_scales;        /* the scale operands as a pointer table, not stacked */
    bool token_order;           /* a_order "token" (else "sorted") */
    bool declared_extent;       /* the pointer table at the plugin's declared [N, K], as the core
                                 * issues a view whose stored blob is [N + 1, K / 16] */
    int  a_group = 64;          /* activation group size */
    bool use_scratch = false;   /* allocate blockmap scratch when non-zero */
    int  N1 = 0, K1 = 0;        /* the odd class's rows and K (0: the even class's) */
    int  parts = 1;             /* stacked parts of the output and of each class's rows */
};

int run_case(const Case& c, unsigned seed) {
    const int Nq[2] = { c.N, c.N1 ? c.N1 : c.N }, Kq[2] = { c.K, c.K1 ? c.K1 : c.K };
    const int N = std::max(Nq[0], Nq[1]), K = std::max(Kq[0], Kq[1]), P = c.parts;
    const int E = c.E, T = c.rows, top_k = c.top_k;
    const int ne0 = (E + 1) / 2, ne1 = E / 2;
    /* the activation: one row per token (token order) or per sorted row (sorted order) */
    const int M = c.token_order ? (T + top_k - 1) / top_k : T;
    const int AG = c.a_group;   /* the activation's scale covers AG of K */
    std::mt19937 rng(seed);
    auto rnd = [&rng](unsigned n) { return (int)(rng() % n); };

    std::vector<uint8_t> codes((size_t)E * N * K);            /* [expert][n][k], 0..15 */
    std::vector<uint8_t> scales((size_t)E * N * (K / 16));    /* E4M3 per (n, 16 of K) */
    std::vector<float>   level((size_t)E);
    for (auto& x : codes) x = (uint8_t)rnd(16);
    for (auto& x : scales) x = (uint8_t)(0x30 + rnd(0x20));   /* E4M3 near 2^-3 .. 2^1 */
    for (auto& x : level) x = 0.5f + 0.25f * (float)rnd(8);

    std::vector<int8_t> act((size_t)M * K);
    for (auto& x : act) x = (int8_t)(rnd(255) - 127);
    std::vector<float> sa((size_t)M * (K / AG));
    for (auto& x : sa) x = 0.03125f * (float)(1 + rnd(16));

    /* the routing: sorted rows split between the experts in order; the last row is dropped */
    std::vector<int32_t> sorted((size_t)T), off((size_t)E + 1, 0);
    for (int i = 0; i < T; ++i)
        sorted[(size_t)i] = i == T - 1 ? -1 : (c.token_order ? rnd((unsigned)(M * top_k)) : i);
    for (int e = 1; e <= E; ++e) off[(size_t)e] = (int32_t)((long long)T * e / E);
    off[(size_t)E] = T - 1;
    for (int e = 1; e <= E; ++e)
        if (off[(size_t)e] < off[(size_t)e - 1]) off[(size_t)e] = off[(size_t)e - 1];

    /* the oracle: a row's expert is the last one whose offset is at most its row */
    std::vector<double> want((size_t)T * N, 0.0);
    for (int i = 0; i < T; ++i) {
        const int flat = sorted[(size_t)i];
        if (flat < 0) continue;
        int e = 0;
        for (int j = 0; j < E; ++j)
            if (off[(size_t)j] <= i) e = j;
        const int tok = c.token_order ? flat / top_k : i;
        const int q = e % 2, fy = N / P, fc = Nq[q] / P;
        for (int col = 0; col < N; ++col) {
            const int pp = col / fy, j = col - pp * fy;
            if (j >= fc) continue;
            const int n = pp * fc + j;
            double acc = 0.0;
            for (int k = 0; k < Kq[q]; ++k) {
                const uint8_t cd = codes[(size_t)((e * N + n) * K + k)];
                const double lv = (cd & 8 ? -kE2m1Mag2[cd & 7] : kE2m1Mag2[cd & 7]) * 0.5;
                const double sc = e4m3_value(scales[(size_t)((e * N + n) * (K / 16) + k / 16)]);
                acc += lv * sc * level[(size_t)e] * act[(size_t)(tok * K + k)] *
                       sa[(size_t)(tok * (K / AG) + k / AG)];
            }
            want[(size_t)(i * N + col)] = acc;
        }
    }

    /* the stored operands: each class's codes as a stacked e2m1 plane, and each expert's scale blob
     * (N rows of K / 16 bytes, then the level in the next row's first four bytes) */
    const int64_t blob[2] = { (int64_t)(Nq[0] + 1) * (Kq[0] / 16), (int64_t)(Nq[1] + 1) * (Kq[1] / 16) };
    std::vector<uint8_t> cw[2], sw[2];
    cw[0].assign((size_t)(ne0 * Nq[0] * Kq[0] / 2) + 1, 0);
    cw[1].assign((size_t)(ne1 * Nq[1] * Kq[1] / 2) + 1, 0);
    sw[0].assign((size_t)(ne0 * blob[0]) + 4, 0);
    sw[1].assign((size_t)(ne1 * blob[1]) + 4, 0);
    for (int e = 0; e < E; ++e) {
        const int q = e % 2, ix = e / 2, nq = Nq[q], kq = Kq[q];
        for (int n = 0; n < nq; ++n)
            for (int k = 0; k < kq; k += 2) {
                const uint8_t lo = codes[(size_t)((e * N + n) * K + k)];
                const uint8_t hi = codes[(size_t)((e * N + n) * K + k + 1)];
                cw[q][(size_t)(ix * nq * kq / 2 + n * kq / 2 + k / 2)] = (uint8_t)(lo | (hi << 4));
            }
        uint8_t* b = sw[q].data() + (size_t)(ix * blob[q]);
        for (int n = 0; n < nq; ++n)
            for (int g = 0; g < kq / 16; ++g)
                b[n * (kq / 16) + g] = scales[(size_t)(e * N * (K / 16) + n * (K / 16) + g)];
        std::memcpy(b + nq * (kq / 16), &level[(size_t)e], sizeof(float));
    }

    /* the scale operands: stacked, or a pointer table of one blob per expert (stride 0) */
    std::vector<void*> ptr[2];
    RadTensor ws[2];
    for (int q = 0; q < 2; ++q) {
        const int64_t ne = q == 0 ? ne0 : (ne1 > 0 ? ne1 : 1);
        if (c.pointer_scales) {
            for (int e = q; e < E; e += 2)
                ptr[q].push_back(sw[q].data() + (size_t)((e / 2) * blob[q]));
            if (ptr[q].empty()) ptr[q].push_back(sw[q].data());
            ws[q] = c.declared_extent ? tensor(ptr[q].data(), RAD_U8, { ne, Nq[q], Kq[q] })
                                      : tensor(ptr[q].data(), RAD_U8, { ne, Nq[q] + 1, Kq[q] / 16 });
            ws[q].stride[0] = 0;
        } else {
            ws[q] = tensor(sw[q].data(), RAD_U8, { ne, Nq[q] + 1, Kq[q] / 16 });
        }
    }
    RadTensor w0 = tensor(cw[0].data(), RAD_FP4E2M1, { ne0, Nq[0], Kq[0] });
    RadTensor w1 = tensor(cw[1].data(), RAD_FP4E2M1, { ne1 > 0 ? ne1 : 1, Nq[1], Kq[1] });
    RadTensor a = tensor(act.data(), RAD_I8, { M, K });
    RadTensor as = tensor(sa.data(), RAD_F32, { M, K / AG });
    RadTensor srt = tensor(sorted.data(), RAD_I32, { T });
    RadTensor ofs = tensor(off.data(), RAD_I32, { E + 1 });
    std::vector<uint16_t> y((size_t)T * N, 0x7fc0);   /* NaN: a column the kernel skips fails */
    RadTensor yt = tensor(y.data(), RAD_BF16, { T, N });

    RadTensor ts[9] = { a, as, w0, ws[0], srt, ofs, w1, ws[1], yt };
    RadParam ps[] = { pint("N", Nq[0]), pint("K", Kq[0]), pint("N_odd", Nq[1]), pint("K_odd", Kq[1]),
                      pint("parts", P), pint("n_expert", E), pint("top_k", top_k),
                      pint("group", AG), pstr("a_order", c.token_order ? "token" : "sorted"),
                      pstr("dtype", "nvfp4a8i8"), pint("M", (T + top_k - 1) / top_k) };
    RadArgs args{};
    args.t = ts;
    args.n_t = 9;
    args.p = ps;
    args.n_p = (int)(sizeof ps / sizeof ps[0]);
    args.world_size = 1;
    std::vector<uint8_t> scratch;
    if (c.use_scratch) {
        const int64_t sb = r4d_rad_moe_gemm_nvfp4_scratch(&args);
        if (sb > 0) scratch.assign((size_t)sb, 0);
        args.scratch = scratch.data();
        args.scratch_bytes = (int64_t)scratch.size();
    }
    const int rc = r4d_rad_moe_gemm_nvfp4a8i8(&args, nullptr);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "    launch: rc %d\n", rc);
        return 1;
    }
    int bad = 0;
    for (int i = 0; i < T * N; ++i) {
        const uint32_t u = (uint32_t)y[(size_t)i] << 16;
        float got;
        std::memcpy(&got, &u, sizeof got);
        const double w = want[(size_t)i];
        if (std::fabs(got - w) > 0.01 * std::fabs(w) + 1e-3) {
            if (bad < 4) std::fprintf(stderr, "    row %d col %d: %g, want %g\n", i / N, i % N, got, w);
            ++bad;
        }
    }
    return bad ? 1 : 0;
}

}  // namespace

int main() {
    const Case cases[] = {
        { 16, 64, 2, 1, 5, false, true },    /* two experts, stacked scales, token order */
        { 16, 64, 2, 1, 5, true, true },     /* the same with a pointer table of scales */
        { 16, 128, 3, 1, 6, false, false },  /* sorted order, three experts */
        { 32, 64, 4, 2, 9, false, true },    /* top 2, four experts */
        { 16, 64, 2, 1, 5, true, true, true },   /* the pointer table as the core issues it */
        { 128, 128, 4, 1, 96, false, false, false, 64, true },   /* scratch-backed WMMA prefill, AG 64 */
        { 128, 128, 4, 2, 96, true, true, true, 128, true },     /* scratch-backed WMMA prefill, pointer, AG 128 */
        { 128, 128, 4, 2, 96, false, true, false, 64, false },   /* uniform WMMA prefill fallback (no scratch) */
        { 64,  128, 4, 2, 96, true, true, true, 128, true },     /* 32x64 WMMA prefill fallback (N=64) */
        { 128, 384, 4, 2, 24, false, true, false, 128, false },  /* FAST256 decode with 256+128 tail */
        { 128, 384, 4, 2, 160, true, true, true, 128, true },    /* 64x128 WMMA prefill with K=384 */
        { 256, 256, 8, 8, 16, true, true, true, 128, true },     /* 32x128 WMMA prefill (T=128 < 48*E) */
        { 128, 128, 8, 8, 16, false, false, false, 64, false },  /* 32x128 uniform fallback, AG 64 */
        /* a rank's half of a parity-split gate_up: [gate | up] of 640 and 512 columns a class,
         * two parts of 640, prefill (128-column tiles) with scratch and without, and decode */
        { 1280, 256, 4, 2, 160, true, true, true, 128, true, 1024, 256, 2 },
        { 1024, 256, 4, 2, 160, false, true, false, 128, false, 1280, 256, 2 },
        { 1280, 256, 4, 2, 24, true, true, true, 128, true, 1024, 256, 2 },
        /* its down projection: K of 640 and 512 a class, one part */
        { 256, 640, 4, 2, 160, true, false, true, 128, true, 256, 512, 1 },
        { 256, 512, 4, 2, 24, false, false, false, 128, false, 256, 640, 1 },
        /* multi-row grouped verify/decode (T > top_k, T <= 64): repeated experts, parts=1 and parts=2 */
        { 128, 256, 4, 8, 64, true, true, true, 128, false },
        { 128, 384, 3, 4, 36, false, false, false, 128, false },
        { 1280, 256, 4, 8, 64, true, true, true, 128, false, 1024, 256, 2 },
    };
    int fails = 0, n = 0;
    for (const Case& c : cases) {
        ++n;
        const int rc = run_case(c, 77u + (unsigned)n);
        if (rc) ++fails;
        std::printf("  case %d (N %d K %d odd %d/%d parts %d experts %d top %d rows %d%s%s%s): %s\n", n,
                    c.N, c.K, c.N1 ? c.N1 : c.N, c.K1 ? c.K1 : c.K, c.parts, c.E,
                    c.top_k, c.rows, c.pointer_scales ? " pointer" : "",
                    c.declared_extent ? " at the declared extent" : "",
                    c.token_order ? "" : " sorted", rc ? "FAIL" : "ok");
    }
    std::printf("%d case(s), %d failure(s)\n", n, fails);
    return fails ? 1 : 0;
}
