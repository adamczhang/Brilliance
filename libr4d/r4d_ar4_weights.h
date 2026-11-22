#pragma once
// r4d_ar4_weights.h: Brilliance (tp4-link-weight), the four-rank host-mailbox two-shot's shard
// weights: where they come from and the tables they give. HIP-free, so the entry
// (r4d_ar_entry.hip), the weighted coded kernel (r4d_ar_wide_tw.hip) and the host test
// (tests/r4d_ar4_weights_test.cpp) derive the same numbers from the same code.
//
// THE PROBLEM. On Windows every four-rank collective crosses host memory, each card over its own
// PCIe link. In the two-shot, rank r reduces a share x_r of the message: hop 1 sends its (1 - x_r)
// of the other shards out and receives 3 x_r back, hop 2 sends its reduced shard to three peers
// (3 x_r) and receives the other (1 - x_r). Each direction of rank r's link therefore carries
// (1 + 2 x_r) N for a message of N bytes, and the slowest such link sets the pace of the step. On
// the test machine the four links run Gen5 x8, Gen5 x8, Gen5 x4 and (bus 32's riser, under load) Gen4 x4: even
// shards (x = 1/4) put 1.5 N on the slowest link, and every prefill step waits on it.
//
// THE RULE (r4d_ar4_weights_from_rates): the fast tier reduces. The ranks whose links are within
// R4D_AR4_EQUAL_RATIO of the fastest share the reduce work evenly (weight 1); every slower rank gets
// none (weight 0) and carries the message once each way instead of 1.5 times. All four in the tier:
// the even split. A tier of one: the even split too (one card would reduce every message, and its
// tables would not fit a slot). Why a tier and not shares in proportion to the rate: the TP4 step
// breakdown on the test machine (tp4-measure, 2026-11-22) found both x4 risers, bus 6 at Gen5 x4 and bus 32 at
// Gen4 x4, about 60-70 ms a prefill step heavier in the host reads and writes than the x8 cards,
// alike, although one has twice the other's rate: what a riser link costs is not in proportion to
// its rate, so a share for the faster riser saves nothing. On the test machine (14, 28, 28, 7 GB/s by rank, buses
// 6, 3, 9, 32) that gives 0,1,1,0: both risers offloaded onto the x8 pair, whether bus 32 trained
// at Gen4 or Gen5.
//
// The weights change which link a word crosses, never its fold (fp32, ascending rank, one rounding),
// and the coded form's codec is lossless, so every table gives the same output bits.
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define R4D_AR4_HD __host__ __device__
#else
#define R4D_AR4_HD
#endif

enum { R4D_AR4_WS = 4 };

// ---- the setting (BRILLIANCE_AR4_SHARD_WEIGHTS) ---------------------------------------------------
// unset or "auto"  from the start-time link probe (r4d_ar4_weights_from_rates); unknown rates: even
// "off", "even"    even shards, the kernels and tables of the build before this setting
// "a,b,c,d"        four weights in rank order (0..R4D_AR4_WEIGHT_MAX, sum above zero)
// "B:w,B:w,..."    weights by PCI bus number (decimal, as the startup report and serve.ps1 -Bus
//                  name the cards); every rank's bus must be named once
enum { R4D_AR4_SPEC_BAD = -1, R4D_AR4_SPEC_AUTO = 0, R4D_AR4_SPEC_EVEN = 1, R4D_AR4_SPEC_RANKS = 2,
       R4D_AR4_SPEC_BUSES = 3 };
enum { R4D_AR4_WEIGHT_MAX = 65535, R4D_AR4_SPEC_MAX_BUSES = 16 };

struct R4DAr4Spec {
  int kind = R4D_AR4_SPEC_AUTO;
  int w[R4D_AR4_WS] = { 0, 0, 0, 0 };                // RANKS
  int n = 0;                                          // BUSES: the pairs
  int bus[R4D_AR4_SPEC_MAX_BUSES] = { 0 };
  int bw[R4D_AR4_SPEC_MAX_BUSES] = { 0 };
};

namespace r4d_ar4_detail {
inline bool same(const char* a, const char* b) { return std::strcmp(a, b) == 0; }
// A decimal 0..max at *s; advances *s. False on no digit or overflow.
inline bool num(const char** s, long long max, int* out) {
  const char* p = *s;
  if (*p < '0' || *p > '9') return false;
  long long v = 0;
  while (*p >= '0' && *p <= '9') {
    v = v * 10 + (*p++ - '0');
    if (v > max) return false;
  }
  *out = (int)v;
  *s = p;
  return true;
}
}  // namespace r4d_ar4_detail

// Parses the setting; null or "" is AUTO. Anything not described above is BAD (no blanks, signs,
// repeats or trailing text).
inline R4DAr4Spec r4d_ar4_spec_parse(const char* s) {
  using namespace r4d_ar4_detail;
  R4DAr4Spec r;
  if (!s || !*s || same(s, "auto")) { r.kind = R4D_AR4_SPEC_AUTO; return r; }
  if (same(s, "off") || same(s, "even")) { r.kind = R4D_AR4_SPEC_EVEN; return r; }
  r.kind = R4D_AR4_SPEC_BAD;
  const bool buses = std::strchr(s, ':') != nullptr;
  const char* p = s;
  long long sum = 0;
  if (!buses) {
    for (int i = 0; i < R4D_AR4_WS; ++i) {
      if (!num(&p, R4D_AR4_WEIGHT_MAX, &r.w[i])) return r;
      sum += r.w[i];
      if (i + 1 < R4D_AR4_WS) {
        if (*p != ',') return r;
        ++p;
      }
    }
    if (*p || sum <= 0) return r;
    r.kind = R4D_AR4_SPEC_RANKS;
    return r;
  }
  for (;;) {
    if (r.n >= R4D_AR4_SPEC_MAX_BUSES) return r;
    int b = 0, w = 0;
    if (!num(&p, 255, &b) || *p != ':') return r;
    ++p;
    if (!num(&p, R4D_AR4_WEIGHT_MAX, &w)) return r;
    for (int i = 0; i < r.n; ++i)
      if (r.bus[i] == b) return r;   // a bus named twice
    r.bus[r.n] = b;
    r.bw[r.n] = w;
    ++r.n;
    sum += w;
    if (!*p) break;
    if (*p != ',') return r;
    ++p;
  }
  if (sum <= 0) return r;
  r.kind = R4D_AR4_SPEC_BUSES;
  return r;
}

// A BUSES spec's weights in rank order, given each rank's bus. False when a rank's bus is unknown
// (negative) or not named, or the named weights of the four ranks sum to zero. A named bus that no
// rank is on is allowed (a profile may name more cards than a start uses).
inline bool r4d_ar4_spec_by_bus(const R4DAr4Spec& s, const int* rank_bus, int* w) {
  long long sum = 0;
  for (int r = 0; r < R4D_AR4_WS; ++r) {
    int found = -1;
    for (int i = 0; i < s.n; ++i)
      if (rank_bus[r] >= 0 && s.bus[i] == rank_bus[r]) found = i;
    if (found < 0) return false;
    w[r] = s.bw[found];
    sum += w[r];
  }
  return sum > 0;
}

inline bool r4d_ar4_weights_equal(const int* w) {
  return w[0] == w[1] && w[1] == w[2] && w[2] == w[3];
}

// ---- the rule ------------------------------------------------------------------------------------
// Links whose rates are within this ratio of the fastest are its tier: a probe scatters by a few
// percent, and PCIe links that really differ do so by a generation or a halving of lanes (2x).
#define R4D_AR4_EQUAL_RATIO 1.25

// Weights from four link rates (GB/s, each direction), as described at the top: 1 for each rank in
// the fastest link's tier, 0 for the rest. False (keep the even split) when a rate is unknown (not
// above zero), when all four are in the tier, or when the tier is one rank.
inline bool r4d_ar4_weights_from_rates(const double* gbs, int* w) {
  double top = 0.0;
  for (int r = 0; r < R4D_AR4_WS; ++r) {
    if (!(gbs[r] > 0.0)) return false;
    if (gbs[r] > top) top = gbs[r];
  }
  int tier = 0;
  for (int r = 0; r < R4D_AR4_WS; ++r) {
    w[r] = R4D_AR4_EQUAL_RATIO * gbs[r] >= top ? 1 : 0;
    tier += w[r];
  }
  return tier >= 2 && tier < R4D_AR4_WS;
}

// One direction of a link's traffic a message under `w` (null: even), in messages: 1 + 2 x for a rank
// reducing a share x (1.5 even, 1 for a rank weighted 0). The startup line reports the slowest
// link's.
inline double r4d_ar4_link_load(const int* w, int r) {
  long long sum = 0;
  for (int s = 0; s < R4D_AR4_WS; ++s) sum += w ? w[s] : 1;
  return 1.0 + 2.0 * (double)(w ? w[r] : 1) / (double)sum;
}

// ---- the tables ----------------------------------------------------------------------------------
// A four-rank two-shot table: shard s is the 16-byte words [off[s], off[s + 1]), reduced by rank s;
// sh_max the longest. The plain form takes r4d_ar_shard_table's (r4d_ar_shards.h); the coded form
// splits whole waves of 32 words the same way, so every shard is whole waves.
struct R4DArwTab { int off[R4D_AR4_WS + 1]; int sh_max; };

// r4d_ar_shard_table at four ranks in waves of `unit` words (1: the plain form, 32: the coded form).
// `w` null: even. n16 must be a multiple of `unit`.
inline void r4d_arw_tab(long long n16, const int* w, int unit, R4DArwTab* t) {
  const long long n = n16 / unit;
  long long len[R4D_AR4_WS];
  if (!w) {
    const long long base = n / R4D_AR4_WS, rem = n % R4D_AR4_WS;
    for (int s = 0; s < R4D_AR4_WS; ++s) len[s] = base + (s < rem ? 1 : 0);
  } else {
    long long sum = 0;
    for (int s = 0; s < R4D_AR4_WS; ++s) sum += w[s];
    long long left = n;
    for (int s = 0; s < R4D_AR4_WS; ++s) {
      len[s] = n * w[s] / sum;
      left -= len[s];
    }
    for (int s = 0; s < R4D_AR4_WS && left > 0; ++s)
      if (w[s] > 0) { ++len[s]; --left; }
  }
  long long mx = 0;
  t->off[0] = 0;
  for (int s = 0; s < R4D_AR4_WS; ++s) {
    t->off[s + 1] = t->off[s] + (int)(len[s] * unit);
    if (len[s] > mx) mx = len[s];
  }
  t->sh_max = (int)(mx * unit);
}

// One coded region of a shard of SL words (r4d_ar_wide_ts.hip's arwc_region_bytes, restated:
// tw.hip checks the two agree): [3 dwords a word][a base byte a group of 8 words][the raw shadow,
// 16 bytes a word], padded to 256 bytes.
R4D_AR4_HD constexpr size_t r4d_arw_coded_shadow_off(long long SL) {
  return ((size_t)12 * SL + (size_t)SL / 8 + 15) & ~(size_t)15;
}
R4D_AR4_HD constexpr size_t r4d_arw_coded_region_bytes(long long SL) {
  return (r4d_arw_coded_shadow_off(SL) + (size_t)16 * SL + 255) & ~(size_t)255;
}

// Whether a table fits a slot of `slot16` words: the plain form's two regions of sh_max words, the
// coded form's two coded regions.
inline bool r4d_arw_tab_fits(const R4DArwTab& t, bool coded, long long slot16) {
  if (coded) return 2 * r4d_arw_coded_region_bytes(t.sh_max) <= (size_t)slot16 * 16;
  return 2LL * t.sh_max <= slot16;
}

// Block b's span [w0, w1) of the longest shard (the kernels' block partition; `unit` 32 rounds the
// span to whole waves as the coded form does), and shard x's end within it.
R4D_AR4_HD inline void r4d_arw_span(int sh_max, int nb, int b, int unit, int* w0, int* w1) {
  int w = (sh_max + nb - 1) / nb;
  if (unit > 1) w = (w + unit - 1) / unit * unit;
  *w0 = b * w;
  int e = *w0 + w;
  if (e > sh_max) e = sh_max;
  *w1 = e;
}
R4D_AR4_HD inline int r4d_arw_lim(const int* off, int w1, int x) {
  const int len = off[x + 1] - off[x];
  return w1 < len ? w1 : len;
}
