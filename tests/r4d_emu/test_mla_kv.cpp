/* tests/r4d_emu: Hy4-flash's kv_b projections on WMMA (r4d_mla_kv_bf16.hip, WP16.3) on the host SIMT
 * emulator.
 *
 *   - mla_project_kv, both modes, against libref's op within one bf16 ulp (a different sum order);
 *   - the law: a token's bytes are the same whatever M, `mb` (every value of the row's tunable),
 *     the chained form or the operand layout (16-byte fragments and global_load_tr_b128, or the
 *     element-by-element path a pitched operand takes) -- every row of every run equal to the
 *     same token run alone, at M 1, through the plain aligned path;
 *   - mla_expand_gate byte-equal to mla_project_kv's expand followed by gate_mul (its fusion
 *     claim, r4d_fuse.cpp), on both paths and every `mb`.
 *
 * The emulator's WMMA sums in its own order (emu.cpp), not the card's: what is checked is that the
 * kernels compute every output from the same fragments in the same chain, which is what the law
 * says; r4d_selftest holds the card's bytes to the same rules. Needs libref (RADIANCE_HOME). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

static const RadKernelInfo* libref_op(const char* op) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        h = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
        if (open && open() < 0) return nullptr;
    }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i)
        if (at(i)->op && !std::strcmp(at(i)->op, op)) return at(i);
    return nullptr;
}

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float u() { return (float)(next() >> 8) / 8388608.0f - 1.0f; }
    float n() { return (u() + u() + u() + u()) * 0.866f; }
};

static RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

static int g_fails = 0, g_cases = 0;
static void ok(bool good, const std::string& what) {
    ++g_cases;
    if (!good) ++g_fails;
    std::printf("%s %s\n", good ? "ok  " : "FAIL", what.c_str());
}

constexpr int64_t kKd = 192, kVd = 256, kL = 512;

/* A [rows, cols] bf16 matrix with row pitch `pitch` (elements) and its first element `off`
 * elements into the allocation: off 1 or an odd pitch puts it off the 16-byte grid. */
struct Mat {
    int64_t rows = 0, cols = 0, pitch = 0, off = 0;
    std::vector<uint16_t> buf;
    Mat(int64_t r, int64_t c, int64_t p, int64_t o, uint16_t fillv = 0x7e7e)
        : rows(r), cols(c), pitch(p), off(o), buf((size_t)(o + r * p + 16), fillv) {}
    uint16_t* data() {
        /* the allocation's own alignment is the vector's: start the matrix on a 16-byte boundary */
        uintptr_t a = (uintptr_t)buf.data();
        const int64_t pad = (int64_t)(((16 - (a & 15)) & 15) / 2);
        return buf.data() + pad + off;
    }
    uint16_t& at(int64_t i, int64_t j) { return data()[i * pitch + j]; }
    RadTensor t() {
        RadTensor x{};
        x.data = data();
        x.dtype = RAD_BF16;
        x.rank = 2;
        x.shape[0] = rows; x.shape[1] = cols;
        x.stride[0] = pitch; x.stride[1] = 1;
        return x;
    }
};

static int run(RadLaunchFn fn, std::vector<RadTensor> t, std::vector<RadParam> p) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
    return fn(&a, nullptr);
}
static std::vector<RadParam> params(int64_t M, int64_t H, const char* mode, int mb, int chain) {
    std::vector<RadParam> p = { pi("M", M), pi("heads", H), pi("k_dim", kKd), pi("v_dim", kVd),
                                pi("latent_dim", kL), ps("dtype", "bf16"), pi("mb", mb), pi("chain", chain) };
    if (mode) p.push_back(ps("mode", mode));
    return p;
}

/* The inputs of one geometry, drawn once: every run copies its rows out of these. */
struct Draw {
    int64_t H, M;
    std::vector<uint16_t> x, w, g;   /* x [M, H*xw], w [H*448, 512], g [M, H*256] */
};
static Draw draw(int64_t H, int64_t M, bool absorb, uint32_t seed) {
    Draw d{ H, M, {}, {}, {} };
    Rng rng(seed);
    const int64_t xw = absorb ? kKd : kL;
    d.x.resize((size_t)(M * H * xw));
    for (auto& v : d.x) v = f2b(rng.n());
    d.w.resize((size_t)(H * (kKd + kVd) * kL));
    for (auto& v : d.w) v = f2b(rng.n() * 0.05f);
    d.g.resize((size_t)(M * H * kVd));
    for (auto& v : d.g) v = f2b(rng.n() * 3.0f);
    return d;
}

/* One run over rows [r0, r0 + n) of the draw; returns y [n, H * yw] packed. `skew` 1 puts every
 * operand off the 16-byte grid (odd pitches, a one-element offset), the element-by-element path. */
static std::vector<uint16_t> project(const Draw& d, bool absorb, bool gate, int64_t r0, int64_t n, int mb,
                                     int chain, bool skew, int* rc) {
    const int64_t H = d.H, xw = absorb ? kKd : kL, yw = absorb ? kL : kVd;
    const int64_t o = skew ? 1 : 0, sp = skew ? 3 : 0;
    Mat x(n, H * xw, H * xw + sp, o), w(H * (kKd + kVd), kL, kL + sp, o), y(n, H * yw, H * yw + sp, o),
        g(n, H * kVd, H * kVd + sp, o);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < H * xw; ++j) x.at(i, j) = d.x[(size_t)((r0 + i) * H * xw + j)];
    for (int64_t i = 0; i < H * (kKd + kVd); ++i)
        for (int64_t j = 0; j < kL; ++j) w.at(i, j) = d.w[(size_t)(i * kL + j)];
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < H * kVd; ++j) g.at(i, j) = d.g[(size_t)((r0 + i) * H * kVd + j)];
    if (gate)
        *rc = run(r4d_rad_mla_expand_gate, { x.t(), w.t(), g.t(), y.t() }, params(n, H, nullptr, mb, chain));
    else
        *rc = run(r4d_rad_mla_project_kv, { x.t(), w.t(), y.t() },
                  params(n, H, absorb ? "absorb" : "expand", mb, chain));
    std::vector<uint16_t> out((size_t)(n * H * yw));
    bool pad_ok = true;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < H * yw; ++j) out[(size_t)(i * H * yw + j)] = y.at(i, j);
        for (int64_t j = H * yw; j < y.pitch; ++j) pad_ok &= y.at(i, j) == 0x7e7e;
    }
    if (!pad_ok) *rc = -999;
    return out;
}

/* gate_mul over a packed [n, H*256] expansion, through the emulated libr4d row (r4d_ihc_bf16.hip) */
static std::vector<uint16_t> gate_chain(const Draw& d, int64_t r0, int64_t n, std::vector<uint16_t> ao,
                                        int* rc) {
    const int64_t W = d.H * kVd;
    Mat a(n, W, W, 0), g(n, W, W, 0), y(n, W, W, 0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < W; ++j) {
            a.at(i, j) = ao[(size_t)(i * W + j)];
            g.at(i, j) = d.g[(size_t)((r0 + i) * W + j)];
        }
    *rc = run(r4d_rad_gate_mul, { a.t(), g.t(), y.t() }, { pi("M", n), pi("n", W), ps("dtype", "bf16") });
    std::vector<uint16_t> out((size_t)(n * W));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < W; ++j) out[(size_t)(i * W + j)] = y.at(i, j);
    return out;
}

/* libref's mla_project_kv on the whole draw, for the bound */
static bool vs_libref(const Draw& d, bool absorb, const std::vector<uint16_t>& got, std::string* why) {
    const RadKernelInfo* ref = libref_op("mla_project_kv");
    if (!ref) { *why = "libref's mla_project_kv not loadable (set RADIANCE_HOME)"; return false; }
    const int64_t H = d.H, M = d.M, xw = absorb ? kKd : kL, yw = absorb ? kL : kVd;
    Mat x(M, H * xw, H * xw, 0), w(H * (kKd + kVd), kL, kL, 0), y(M, H * yw, H * yw, 0);
    std::memcpy(x.data(), d.x.data(), d.x.size() * 2);
    std::memcpy(w.data(), d.w.data(), d.w.size() * 2);
    if (run(ref->launch, { x.t(), w.t(), y.t() }, params(M, H, absorb ? "absorb" : "expand", 0, 0)) != RAD_OK) {
        *why = "libref refused";
        return false;
    }
    int64_t bad = 0;
    for (int64_t i = 0; i < M * H * yw; ++i) {
        const double e = b2f(y.data()[i]), g = b2f(got[(size_t)i]);
        if (!std::isfinite(g) || std::fabs(g - e) > 2e-5 + 0.0079 * std::fabs(e)) {
            if (!bad) {
                char b[160];
                std::snprintf(b, sizeof b, "element %lld got %.9g want %.9g", (long long)i, g, e);
                *why = b;
            }
            ++bad;
        }
    }
    return bad == 0;
}

int main() {
    struct G { int64_t H, M; };
    for (const bool absorb : { true, false })
        for (const G& g : { G{ 1, 1 }, G{ 2, 17 }, G{ 2, 37 }, G{ 1, 70 } }) {
            const char* mode = absorb ? "absorb" : "expand";
            const Draw d = draw(g.H, g.M, absorb, (uint32_t)(g.H * 100 + g.M + (absorb ? 0 : 7)));
            const int64_t yw = (absorb ? kL : kVd) * g.H;
            int rc = 0;
            /* the reference bytes: every token alone, aligned, plain, the law's mb for one row */
            std::vector<uint16_t> alone((size_t)(g.M * yw));
            bool rc_ok = true;
            for (int64_t t = 0; t < g.M; ++t) {
                const auto r = project(d, absorb, false, t, 1, 0, 0, false, &rc);
                rc_ok &= rc == RAD_OK;
                std::memcpy(&alone[(size_t)(t * yw)], r.data(), r.size() * 2);
            }
            char w[200];
            std::snprintf(w, sizeof w, "%s heads %lld: every token alone launches", mode, (long long)g.H);
            ok(rc_ok, w);
            std::string why;
            std::snprintf(w, sizeof w, "%s heads %lld M %lld within one bf16 ulp of libref", mode,
                          (long long)g.H, (long long)g.M);
            ok(vs_libref(d, absorb, alone, &why), std::string(w) + (why.empty() ? "" : ": " + why));
            for (const bool skew : { false, true })
                for (const int chain : { 0, 1 })
                    for (const int mb : { 0, 1, 2, 4, 8 }) {
                        if (skew && (chain || mb > 1)) continue;   /* one form off the grid */
                        const auto all = project(d, absorb, false, 0, g.M, mb, chain, skew, &rc);
                        std::snprintf(w, sizeof w,
                                      "%s heads %lld M %lld mb %d%s%s: every row's bytes equal the row alone",
                                      mode, (long long)g.H, (long long)g.M, mb, chain ? " chained" : "",
                                      skew ? " off the 16-byte grid" : "");
                        ok(rc == RAD_OK && all == alone, w);
                        if (absorb) continue;
                        /* the fused gate against the chain it replaces, on these bytes */
                        const auto fused = project(d, false, true, 0, g.M, mb, chain, skew, &rc);
                        int rc2 = 0;
                        const auto chained = gate_chain(d, 0, g.M, alone, &rc2);
                        std::snprintf(w, sizeof w,
                                      "mla_expand_gate heads %lld M %lld mb %d%s%s == expand + gate_mul, bit for bit",
                                      (long long)g.H, (long long)g.M, mb, chain ? " chained" : "",
                                      skew ? " off the 16-byte grid" : "");
                        ok(rc == RAD_OK && rc2 == RAD_OK && fused == chained, w);
                    }
            /* a slice that starts mid-matrix: rows [5, M) as their own launch */
            if (g.M > 5) {
                const auto tail = project(d, absorb, false, 5, g.M - 5, 0, 0, false, &rc);
                std::snprintf(w, sizeof w, "%s heads %lld rows 5..%lld as a launch of their own: the same bytes",
                              mode, (long long)g.H, (long long)g.M);
                ok(rc == RAD_OK && std::equal(tail.begin(), tail.end(), alone.begin() + 5 * yw), w);
            }
        }
    std::printf("%d case(s), %d failure(s)\n", g_cases, g_fails);
    return g_fails ? 1 : 0;
}
