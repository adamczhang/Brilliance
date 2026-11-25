/* hy4_dfly_cases.h -- dfly_sel operands for the cross-library checks (libavx's avx_check,
 * tests/r4d_emu/test_hy4.cpp, r4d_selftest): the golden cases of hy4_dfly_golden.h laid out as
 * tests/ref_hy4_ops_test.cpp lays them out, and seeded random geometries. Header-only, no ABI types:
 * each harness copies the integer planes into its own operand type and holds the library under test
 * to libref on the same bytes.
 *
 * Golden layout: one sequence a case, a shuffled page map per sequence (page ids unique across
 * sequences), the source at snapshot row 2s+1 with another row's data in row 2s, the recalled pages
 * shuffled and the own page last (qsa_select's form with `own` 1). */
#pragma once
#include "hy4_dfly_golden.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace hy4dfly {

struct Case {
    int64_t M = 0, S = 0, R = 0, G = 0, pitch = 0, blocks = 0, ratio = 4;
    std::vector<int64_t> sel, nsel, src, bt, pos, cu;   /* [R,G] [R] [S] [S,pitch] [M] [S+1] */
};

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 33; }
    int64_t below(int64_t n) { return n > 0 ? (int64_t)(next() % (uint64_t)n) : 0; }
};

/* The golden cases `idx` in one call. */
inline Case golden(const std::vector<int>& idx, uint64_t seed) {
    Case c;
    c.S = (int64_t)idx.size();
    c.G = kDflyCase[idx[0] * 7];
    c.blocks = c.G + 2;
    for (int i : idx) {
        const int32_t* k = &kDflyCase[i * 7];
        c.M += k[2];
        c.pitch = std::max<int64_t>(c.pitch, (k[1] + k[2]) / c.ratio + 3);
    }
    c.R = 2 * c.S;
    Lcg rng(seed);
    c.sel.assign((size_t)(c.R * c.G), -7);
    c.nsel.assign((size_t)c.R, 99);
    c.src.assign((size_t)c.S, 0);
    c.bt.assign((size_t)(c.S * c.pitch), 0);
    c.pos.assign((size_t)c.M, 0);
    c.cu.assign((size_t)c.S + 1, 0);
    int64_t r = 0;
    for (int64_t s = 0; s < c.S; ++s) {
        const int32_t* k = &kDflyCase[idx[(size_t)s] * 7];
        const int64_t a = k[1], B = k[2], g0 = (a - 1) / c.ratio;
        std::vector<int64_t> perm((size_t)c.pitch);
        for (int64_t b = 0; b < c.pitch; ++b) perm[(size_t)b] = b;
        for (int64_t b = c.pitch - 1; b > 0; --b) std::swap(perm[(size_t)b], perm[(size_t)rng.below(b + 1)]);
        for (int64_t b = 0; b < c.pitch; ++b) c.bt[(size_t)(s * c.pitch + b)] = perm[(size_t)b] * c.S + s;
        const int64_t kr = 2 * s + 1;
        c.src[(size_t)s] = kr;
        c.nsel[(size_t)(2 * s)] = 1 + rng.below(3);
        if (k[3] == 0) {
            c.nsel[(size_t)kr] = 0;
        } else {
            std::vector<int64_t> rec(&kDflyRec[k[4]], &kDflyRec[k[4]] + k[5]);
            for (int64_t i = (int64_t)rec.size() - 1; i > 0; --i) std::swap(rec[(size_t)i], rec[(size_t)rng.below(i + 1)]);
            for (size_t i = 0; i < rec.size(); ++i) c.sel[(size_t)(kr * c.G) + i] = perm[(size_t)rec[i]] * c.S + s;
            c.sel[(size_t)(kr * c.G) + rec.size()] = perm[(size_t)g0] * c.S + s;
            c.nsel[(size_t)kr] = (int64_t)rec.size() + 1;
        }
        c.cu[(size_t)s] = r;
        for (int64_t j = 0; j < B; ++j) c.pos[(size_t)r++] = a + j;
    }
    c.cu[(size_t)c.S] = r;
    return c;
}

/* The golden cases of one G, all in one call. */
inline std::vector<int> golden_of_g(int64_t G) {
    std::vector<int> idx;
    for (int i = 0; i < kDflyCases; ++i)
        if (kDflyCase[i * 7] == G) idx.push_back(i);
    return idx;
}

/* A random geometry: 1..4 sequences of 0..6 rows, ratio 1..5, anchors 0..80, a source that is
 * selecting (any count 1..blocks_in+1), dense, -1 or past the snapshot, rows before cu[0] and past
 * cu[S], `blocks` at the bound or up to two above it. `overflow` also lets a table outrun `blocks`
 * and a count outrun `blocks_in` (those rows fail to -1, len 0; a debug libref also returns
 * RAD_E_SHAPE, after writing them). */
inline Case random(uint64_t seed, bool overflow = false) {
    Lcg rng(seed * 2654435761ull + 17);
    Case c;
    c.ratio = 1 + rng.below(5);
    c.S = 1 + rng.below(4);
    c.G = 1 + rng.below(12);
    c.R = c.S + 1 + rng.below(4);
    const int64_t lead = rng.below(2), tail = rng.below(2);
    std::vector<int64_t> B((size_t)c.S), A((size_t)c.S);
    int64_t need = 1;
    for (int64_t s = 0; s < c.S; ++s) {
        B[(size_t)s] = rng.below(5) == 0 ? 0 : 1 + rng.below(6);
        A[(size_t)s] = rng.below(8) == 0 ? 0 : rng.below(81);
        const int64_t g0 = A[(size_t)s] > 0 ? (A[(size_t)s] - 1) / c.ratio : 0;
        const int64_t e = B[(size_t)s] ? (A[(size_t)s] + B[(size_t)s] - 1) / c.ratio - g0 : 0;
        c.pitch = std::max<int64_t>(c.pitch, (A[(size_t)s] + B[(size_t)s]) / c.ratio + 2);
        need = std::max<int64_t>(need, std::max<int64_t>(c.G, g0) + e + 1);
    }
    c.blocks = overflow ? std::max<int64_t>(1, need - rng.below(3)) : need + rng.below(3);
    c.M = lead + tail;
    for (int64_t b : B) c.M += b;
    if (c.M == 0) c.M = 1;
    c.sel.assign((size_t)(c.R * c.G), 0);
    for (auto& v : c.sel) v = rng.below(c.S * c.pitch + 3) - 1;
    c.nsel.assign((size_t)c.R, 0);
    /* overflow: a count up to blocks_in + 2, one recalled page more than `sel` holds. */
    for (auto& v : c.nsel) v = rng.below(3) == 0 ? 0 : 1 + rng.below(c.G + (overflow ? 2 : 1));
    c.src.assign((size_t)c.S, 0);
    for (auto& v : c.src) {
        const int64_t u = rng.below(10);
        v = u == 0 ? -1 : u == 1 ? c.R + rng.below(3) : rng.below(c.R);
    }
    c.bt.assign((size_t)(c.S * c.pitch), 0);
    for (auto& v : c.bt) v = rng.below(c.S * c.pitch * 2);
    c.pos.assign((size_t)c.M, 0);
    for (auto& v : c.pos) v = rng.below(90);        /* rows no sequence holds keep garbage */
    c.cu.assign((size_t)c.S + 1, 0);
    int64_t r = lead;
    for (int64_t s = 0; s < c.S; ++s) {
        c.cu[(size_t)s] = r;
        for (int64_t j = 0; j < B[(size_t)s]; ++j) c.pos[(size_t)r++] = A[(size_t)s] + j;
    }
    c.cu[(size_t)c.S] = r;
    return c;
}

}  // namespace hy4dfly
