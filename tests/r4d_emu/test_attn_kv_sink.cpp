/* Drives libr4d's attn_kv_sink_h256 (r4d_attn_kv_sink_h256.hip, Hy4-flash's non-absorbed prefill)
 * through the SIMT emulator and checks every output element against an f64 port of libref's
 * ref_attn_kv_sink: ragged batches of (query rows, cached prefix) at window 0 and 512, heads at
 * 1..64 (a grid axis, so most cases keep it small), sinks that matter and one that swamps the
 * scores. Also:
 *   - causality: changing the keys and values after a row's own position leaves its bytes;
 *   - the absorbed form: keys and values expanded from a latent through per-head matrices (bf16,
 *     as kv_b and mla_kv_split produce them) against the f64 latent-space attention the absorbed
 *     kernel computes, within 5e-3 of a row's norm (PLAN WP6c);
 *   - a malformed sequence (n_s < m_s) writes zero rows, and the launch's refusals. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int D = 256;
std::mt19937 rng(4242);
int fails = 0;

float bf(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t tobf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float uni(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}
RadParam pint(const char* k, int64_t v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam pf64(const char* k, double v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_F64;
    p.dval = v;
    return p;
}
RadParam pstr(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

void check(bool c, const std::string& what) {
    std::printf("  %s  %s\n", c ? "ok  " : "FAIL", what.c_str());
    if (!c) ++fails;
}

/* The operands of one launch, dense [rows, heads, 256] planes (16-byte aligned by the vector). */
struct Op {
    int H = 0, window = 0, M = 0, N = 0;
    std::vector<int> qlen, pre;
    std::vector<uint16_t> q, k, v, out;
    std::vector<int32_t> cuq, cuk;
    std::vector<float> sinks;
    void shape(int heads, int win, std::vector<int> ql, std::vector<int> pr) {
        H = heads; window = win; qlen = ql; pre = pr;
        M = N = 0;
        cuq = { 0 };
        cuk = { 0 };
        for (size_t s = 0; s < ql.size(); ++s) {
            M += ql[s];
            N += ql[s] + pr[s];
            cuq.push_back(M);
            cuk.push_back(N);
        }
        q.assign((size_t)M * H * D, 0);
        k.assign((size_t)N * H * D, 0);
        v.assign((size_t)N * H * D, 0);
        out.assign((size_t)M * H * D, 0xffff);
        sinks.assign((size_t)H, 0.0f);
    }
    int run() {
        RadTensor t[7] = { tens(q.data(), RAD_BF16, { M, H, D }), tens(k.data(), RAD_BF16, { N, H, D }),
                           tens(v.data(), RAD_BF16, { N, H, D }), tens(cuq.data(), RAD_I32, { (int64_t)cuq.size() }),
                           tens(cuk.data(), RAD_I32, { (int64_t)cuk.size() }), tens(sinks.data(), RAD_F32, { H }),
                           tens(out.data(), RAD_BF16, { M, H * D }) };
        RadParam p[] = { pint("M", M), pint("N", N), pint("heads", H), pint("head_dim", D),
                         pint("window", window), pf64("scale", 1.0 / 16.0), pstr("dtype", "bf16") };
        RadArgs a{};
        a.t = t;
        a.n_t = 7;
        a.p = p;
        a.n_p = 7;
        a.world_size = 1;
        return r4d_rad_attn_kv_sink_h256(&a, nullptr);
    }
};

/* libref's ref_attn_kv_sink in f64 on the bf16 operands: row r, head h, the 256 outputs */
void reference(const Op& o, std::vector<double>& want) {
    want.assign((size_t)o.M * o.H * D, 0.0);
    for (size_t s = 0; s + 1 < o.cuq.size(); ++s) {
        const int m = o.cuq[s + 1] - o.cuq[s], n = o.cuk[s + 1] - o.cuk[s];
        for (int i = 0; i < m; ++i) {
            const int r = o.cuq[s] + i, p = n - m + i;
            const int lo = o.window > 0 ? std::max(0, p - o.window + 1) : 0;
            for (int h = 0; h < o.H; ++h) {
                std::vector<double> sc((size_t)(p - lo + 1));
                double mx = o.sinks[(size_t)h];
                for (int j = lo; j <= p; ++j) {
                    double dot = 0;
                    const size_t qo = ((size_t)r * o.H + h) * D, ko = ((size_t)(o.cuk[s] + j) * o.H + h) * D;
                    for (int c = 0; c < D; ++c) dot += (double)bf(o.q[qo + c]) * bf(o.k[ko + c]);
                    sc[(size_t)(j - lo)] = dot / 16.0;
                    mx = std::max(mx, sc[(size_t)(j - lo)]);
                }
                double den = std::exp(o.sinks[(size_t)h] - mx);
                double* w = &want[((size_t)r * o.H + h) * D];
                for (int j = lo; j <= p; ++j) {
                    const double e = std::exp(sc[(size_t)(j - lo)] - mx);
                    den += e;
                    const size_t vo = ((size_t)(o.cuk[s] + j) * o.H + h) * D;
                    for (int c = 0; c < D; ++c) w[c] += e * bf(o.v[vo + c]);
                }
                for (int c = 0; c < D; ++c) w[c] /= den;
            }
        }
    }
}

/* elementwise bound and the worst row's relative error (norm of the difference over the norm) */
void compare(const Op& o, const std::vector<double>& want, const std::string& name) {
    bool elem = true;
    double worst = 0, max_abs = 0;
    for (int r = 0; r < o.M; ++r)
        for (int h = 0; h < o.H; ++h) {
            double num = 0, den = 0;
            for (int c = 0; c < D; ++c) {
                const size_t e = ((size_t)r * o.H + h) * D + c;
                const double g = bf(o.out[e]), w = want[e], d = std::fabs(g - w);
                elem = elem && std::isfinite(g) && d <= 2e-3 + 8e-3 * std::fabs(w);
                max_abs = std::max(max_abs, d);
                num += d * d;
                den += w * w;
            }
            if (den > 0) worst = std::max(worst, std::sqrt(num / den));
        }
    char buf[160];
    std::snprintf(buf, sizeof buf, " (max |err| %.3g, worst row %.3g)", max_abs, worst);
    check(elem && worst <= 5e-3, name + buf);
}

void fill_random(Op& o, float qa, float ka, float va) {
    for (auto& x : o.q) x = tobf(uni(qa));
    for (auto& x : o.k) x = tobf(uni(ka));
    for (auto& x : o.v) x = tobf(uni(va));
    for (int h = 0; h < o.H; ++h) o.sinks[(size_t)h] = uni(4.0f);
}

void case_random(const std::string& name, int H, int window, std::vector<int> ql, std::vector<int> pr,
                 float swamp = 0.0f) {
    Op o;
    o.shape(H, window, ql, pr);
    fill_random(o, 2.0f, 2.0f, 4.0f);
    if (swamp != 0.0f) o.sinks[0] = swamp;
    const int rc = o.run();
    if (rc != RAD_OK) {
        check(false, name + " rc " + std::to_string(rc));
        return;
    }
    std::vector<double> want;
    reference(o, want);
    compare(o, want, name);

    /* causality: scramble every sequence's keys and values after its first row's position */
    const std::vector<uint16_t> first = o.out;
    for (size_t s = 0; s + 1 < o.cuq.size(); ++s)
        for (int j = o.pre[s] + 1; j < o.cuk[s + 1] - o.cuk[s]; ++j)
            for (int e = 0; e < H * D; ++e) {
                const size_t x = (size_t)(o.cuk[s] + j) * H * D + e;
                o.k[x] = tobf(uni(3.0f));
                o.v[x] = tobf(uni(9.0f));
            }
    o.run();
    bool same = true;
    for (size_t s = 0; s + 1 < o.cuq.size(); ++s)
        for (int e = 0; e < H * D; ++e) {
            const size_t x = (size_t)o.cuq[s] * H * D + e;
            same = same && o.out[x] == first[x];
        }
    check(same, name + ": later keys leave the first row's bytes");
}

/* Keys and values from a latent: k = [W_UK c | rope], v = W_UV c, rounded to bf16 as kv_b's GEMM
 * and mla_kv_split leave them, against the f64 absorbed attention on the same latent, weights and
 * query: score = scale (q_nope . W_UK c + q_rope . rope), out = W_UV (sum p c) / den. */
void case_absorbed(int H, int window, std::vector<int> ql, std::vector<int> pr) {
    constexpr int L = 512, KD = 192, R = 64;
    Op o;
    o.shape(H, window, ql, pr);
    std::vector<float> c((size_t)o.N * L), rope((size_t)o.N * R), wuk((size_t)H * KD * L),
        wuv((size_t)H * D * L);
    for (auto& x : c) x = bf(tobf(uni(1.0f)));
    for (auto& x : rope) x = bf(tobf(uni(1.0f)));
    for (auto& x : wuk) x = bf(tobf(uni(0.08f)));
    for (auto& x : wuv) x = bf(tobf(uni(0.08f)));
    for (auto& x : o.q) x = tobf(uni(2.0f));
    for (int h = 0; h < H; ++h) o.sinks[(size_t)h] = uni(3.0f);
    for (int i = 0; i < o.N; ++i)
        for (int h = 0; h < H; ++h) {
            uint16_t* kr = &o.k[((size_t)i * H + h) * D];
            uint16_t* vr = &o.v[((size_t)i * H + h) * D];
            for (int d = 0; d < KD; ++d) {
                float a = 0;
                for (int j = 0; j < L; ++j) a += wuk[((size_t)h * KD + d) * L + j] * c[(size_t)i * L + j];
                kr[d] = tobf(a);
            }
            for (int d = 0; d < R; ++d) kr[KD + d] = tobf(rope[(size_t)i * R + d]);
            for (int d = 0; d < D; ++d) {
                float a = 0;
                for (int j = 0; j < L; ++j) a += wuv[((size_t)h * D + d) * L + j] * c[(size_t)i * L + j];
                vr[d] = tobf(a);
            }
        }
    const int rc = o.run();
    std::string name = "absorbed form, heads " + std::to_string(H) + " window " + std::to_string(window);
    if (rc != RAD_OK) {
        check(false, name + " rc " + std::to_string(rc));
        return;
    }
    std::vector<double> want((size_t)o.M * H * D, 0.0);
    for (size_t s = 0; s + 1 < o.cuq.size(); ++s) {
        const int m = o.cuq[s + 1] - o.cuq[s], n = o.cuk[s + 1] - o.cuk[s];
        for (int i = 0; i < m; ++i) {
            const int r = o.cuq[s] + i, p = n - m + i, lo = window > 0 ? std::max(0, p - window + 1) : 0;
            for (int h = 0; h < H; ++h) {
                const uint16_t* qr = &o.q[((size_t)r * H + h) * D];
                std::vector<double> qa(L, 0.0), acc(L, 0.0), sc((size_t)(p - lo + 1));
                for (int j = 0; j < L; ++j)
                    for (int d = 0; d < KD; ++d) qa[j] += (double)bf(qr[d]) * wuk[((size_t)h * KD + d) * L + j];
                double mx = o.sinks[(size_t)h];
                for (int t = lo; t <= p; ++t) {
                    const size_t key = (size_t)(o.cuk[s] + t);
                    double dot = 0;
                    for (int j = 0; j < L; ++j) dot += qa[j] * c[key * L + j];
                    for (int d = 0; d < R; ++d) dot += (double)bf(qr[KD + d]) * rope[key * R + d];
                    sc[(size_t)(t - lo)] = dot / 16.0;
                    mx = std::max(mx, sc[(size_t)(t - lo)]);
                }
                double den = std::exp(o.sinks[(size_t)h] - mx);
                for (int t = lo; t <= p; ++t) {
                    const double e = std::exp(sc[(size_t)(t - lo)] - mx);
                    den += e;
                    for (int j = 0; j < L; ++j) acc[j] += e * c[(size_t)(o.cuk[s] + t) * L + j];
                }
                double* w = &want[((size_t)r * H + h) * D];
                for (int d = 0; d < D; ++d) {
                    double a = 0;
                    for (int j = 0; j < L; ++j) a += wuv[((size_t)h * D + d) * L + j] * acc[j];
                    w[d] = a / den;
                }
            }
        }
    }
    double worst = 0;
    for (int r = 0; r < o.M; ++r)
        for (int h = 0; h < H; ++h) {
            double num = 0, den = 0;
            for (int d = 0; d < D; ++d) {
                const size_t e = ((size_t)r * H + h) * D + d;
                num += (bf(o.out[e]) - want[e]) * (bf(o.out[e]) - want[e]);
                den += want[e] * want[e];
            }
            if (den > 0) worst = std::max(worst, std::sqrt(num / den));
        }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (worst row %.3g)", worst);
    check(worst <= 5e-3, name + buf);
}

void case_malformed() {
    Op o;
    o.shape(2, 0, { 20, 7 }, { 0, 3 });
    fill_random(o, 1.0f, 1.0f, 1.0f);
    o.cuk = { 0, 10, 30 };   /* sequence 0: 10 keys for 20 rows */
    const int rc = o.run();
    bool zero0 = true, set1 = true;
    for (int e = 0; e < 20 * 2 * D; ++e) zero0 = zero0 && o.out[(size_t)e] == 0;
    for (int e = 20 * 2 * D; e < 27 * 2 * D; ++e) set1 = set1 && o.out[(size_t)e] != 0xffff;
    check(rc == RAD_OK && zero0 && set1, "n_s < m_s: that sequence's rows are zero, the next written");
}

void case_refusals() {
    Op o;
    o.shape(2, 0, { 3 }, { 0 });
    auto launch = [&](auto&& edit) {
        RadTensor t[7] = { tens(o.q.data(), RAD_BF16, { o.M, 2, D }), tens(o.k.data(), RAD_BF16, { o.N, 2, D }),
                           tens(o.v.data(), RAD_BF16, { o.N, 2, D }), tens(o.cuq.data(), RAD_I32, { 2 }),
                           tens(o.cuk.data(), RAD_I32, { 2 }), tens(o.sinks.data(), RAD_F32, { 2 }),
                           tens(o.out.data(), RAD_BF16, { o.M, 2, D }) };
        std::vector<RadParam> p = { pint("M", o.M), pint("N", o.N), pint("heads", 2), pint("head_dim", D),
                                    pint("window", 0), pf64("scale", 1.0 / 16.0), pstr("dtype", "bf16") };
        edit(t, p);
        RadArgs a{};
        a.t = t;
        a.n_t = 7;
        a.p = p.data();
        a.n_p = (int)p.size();
        a.world_size = 1;
        return r4d_rad_attn_kv_sink_h256(&a, nullptr);
    };
    check(launch([](RadTensor*, std::vector<RadParam>&) {}) == RAD_OK, "refusals: the plain launch runs");
    check(launch([](RadTensor*, std::vector<RadParam>& p) { p[3].ival = 128; }) == RAD_E_SHAPE,
          "refusals: head_dim 128");
    check(launch([](RadTensor* t, std::vector<RadParam>&) { t[2].dtype = RAD_F16; }) == RAD_E_DTYPE,
          "refusals: f16 values");
    check(launch([](RadTensor* t, std::vector<RadParam>&) { t[5].shape[0] = 3; }) == RAD_E_SHAPE,
          "refusals: three sinks for two heads");
    check(launch([](RadTensor* t, std::vector<RadParam>&) { t[1].stride[0] = 2 * D + 4; }) == RAD_E_ALIGN,
          "refusals: a key pitch off 16 bytes");
    check(launch([](RadTensor* t, std::vector<RadParam>&) { t[4].shape[0] = 3; }) == RAD_E_SHAPE,
          "refusals: cu_k longer than cu_q");
}

}  // namespace

int main() {
    std::printf("attn_kv_sink_h256 on the emulator\n");
    case_refusals();
    case_random("one row, no prefix, window 512", 4, 512, { 1 }, { 0 });
    case_random("16 rows, no prefix, window 0", 3, 0, { 16 }, { 0 });
    case_random("64 rows after 511, window 512", 2, 512, { 64 }, { 511 });
    case_random("100 rows after 37, window 512, swamping sink", 2, 512, { 100 }, { 37 }, 40.0f);
    case_random("ragged: 1+0, 16+0, 64+511, 100+37, window 512", 2, 512, { 1, 16, 64, 100 }, { 0, 0, 511, 37 });
    case_random("ragged: 1+0, 16+0, 64+511, 100+37, window 0", 2, 0, { 1, 16, 64, 100 }, { 0, 0, 511, 37 });
    case_random("window edge: 513 rows, window 512", 1, 512, { 513 }, { 0 });
    case_random("prefix 2047, window 0 and 3 heads", 3, 0, { 2 }, { 2047 });
    case_random("prefix 2047, window 512", 1, 512, { 1 }, { 2047 });
    case_random("64 heads", 64, 512, { 20 }, { 5 });
    case_absorbed(2, 512, { 40 }, { 30 });
    case_absorbed(2, 0, { 17, 50 }, { 0, 90 });
    case_random("1024 rows after 511, window 512", 1, 512, { 1024 }, { 511 });
    case_random("300 rows after 1748, window 0", 1, 0, { 300 }, { 1748 });
    case_malformed();
    std::printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
