/* Drives libr4d's absorbed MLA with a sink (r4d_mla_sink.hip: mla_attn_paged_sink and
 * mla_attn_sel_sink) through the SIMT emulator. Every output element against an f64 softmax with
 * the sink in its denominator; the launch forms (16 or 32 heads a workgroup, element-wise key load)
 * against each other bit for bit; the unsplit form within the bound; verify rows against the
 * single-row decode at each position, bit for bit; the paged form at window 0 against the
 * selection form over the same pages, bit for bit; the window edge with boosted keys on both sides
 * of it; sinks far above and below the scores; unmapped and out-of-range pages; heads 8 to 64.
 * The stable law (`stable 1`, WP16 perf.md 3.4.1): a row's bytes in launches of 2 to 64 rows of 1,
 * 3 and 8 sequences equal its one-row launch; paged window 0 == sel past the old split cap; the
 * lengths 0..2049 (and past the law's last slot) at both windows; HG 1 / 2 / 4 equal; the table's
 * width moves no bit; the fused form (one workgroup a row folding the slots) equals the split form;
 * no room for the partials runs the fused form, no room for its accumulator is refused, never
 * unsplit. The split cap: `stable 1` splits to 48 rows (the verify heights, 8 streams x 6), which
 * the scratch hook covers and the launch writes the parts for, and fuses past them; `stable 0`
 * keeps its 16 / 32 rows. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" void r4d_mla_sink_pin_form(int form);

namespace {

constexpr int L = 512, R = 64, W = L + R;
constexpr double kScale = 0.0625;
/* What the scratch holds for a row and head in either form: the 512 latent values it accumulates and
 * after them the slot's maximum and denominator (r4d_mla_sink.hip's part, padded to 516). */
constexpr int64_t kAccWords = 516;

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

/* One launch's operands. Paged: sequences with query lengths and contexts, one table each. The
 * selection form reads `sel_bt` / `sel_len`, one table row a query row. */
struct Problem {
    int heads = 8, bs = 4, window = 0, pages = 0, blocks = 0;
    std::vector<int> qlen, ctx;
    std::vector<int> bt, used, cu;          /* paged */
    std::vector<int> sel_bt, sel_len;       /* selection */
    int sel_blocks = 0;
    std::vector<uint16_t> q, qr, kv;
    std::vector<float> sinks;
    int rows() const { return cu.empty() ? 0 : cu.back(); }
};

uint16_t bf(float v) { return rad_f32_to_bf16(v); }
float fb(uint16_t v) { return rad_bf16_to_f32(v); }

/* Pages are a scramble of the sequence's blocks; `holes` unmaps a block and points one past the
 * cache. */
void make_paged(Problem& p, std::vector<int> qlen, std::vector<int> ctx, int heads, int bs,
                int window, bool holes, std::mt19937& rng, float qamp = 0.5f) {
    p.heads = heads;
    p.bs = bs;
    p.window = window;
    p.qlen = qlen;
    p.ctx = ctx;
    const int S = (int)qlen.size();
    p.blocks = 1;
    int total = 0;
    for (int s = 0; s < S; ++s) {
        p.blocks = std::max(p.blocks, (ctx[s] + bs - 1) / bs);
        total += (ctx[s] + bs - 1) / bs;
    }
    p.pages = std::max(total, 1) + 3;
    std::vector<int> order(p.pages);
    for (int i = 0; i < p.pages; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    p.bt.assign((size_t)S * p.blocks, -1);
    int next = 0;
    for (int s = 0; s < S; ++s)
        for (int b = 0; b < (ctx[s] + bs - 1) / bs; ++b) p.bt[(size_t)s * p.blocks + b] = order[next++];
    if (holes)
        for (int s = 0; s < S; ++s) {
            const int nb = (ctx[s] + bs - 1) / bs;
            if (nb > 3) p.bt[(size_t)s * p.blocks + 1] = -1;
            if (nb > 6) p.bt[(size_t)s * p.blocks + nb - 3] = p.pages + 5;
        }
    p.used = ctx;
    p.cu.assign(S + 1, 0);
    for (int s = 0; s < S; ++s) p.cu[s + 1] = p.cu[s] + qlen[s];
    const int M = p.rows();
    std::normal_distribution<float> nrm(0.f, 1.f);
    p.q.resize((size_t)M * heads * L);
    p.qr.resize((size_t)M * heads * R);
    p.kv.resize((size_t)p.pages * bs * W);
    for (auto& v : p.q) v = bf(qamp * nrm(rng));
    for (auto& v : p.qr) v = bf(qamp * nrm(rng));
    for (auto& v : p.kv) v = bf(nrm(rng));
    /* sinks: ordinary, far above (30, 100), far below, and the ordinary scale again */
    const float table[] = { 0.5f, 30.f, -30.f, 2.f, 100.f, -1.f, 4.f, -100.f };
    p.sinks.resize(heads);
    for (int h = 0; h < heads; ++h) p.sinks[h] = table[h % 8] + 0.01f * h;
}

/* Selection tables equal to each row's sequence table, length p + 1: window 0's keys. */
void make_sel_from_paged(Problem& p) {
    const int M = p.rows(), S = (int)p.qlen.size();
    p.sel_blocks = p.blocks;
    p.sel_bt.assign((size_t)M * p.blocks, -1);
    p.sel_len.assign(M, 0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < p.qlen[s]; ++i) {
            const int r = p.cu[s] + i;
            std::copy(p.bt.begin() + (size_t)s * p.blocks, p.bt.begin() + (size_t)(s + 1) * p.blocks,
                      p.sel_bt.begin() + (size_t)r * p.blocks);
            p.sel_len[r] = p.used[s] - p.qlen[s] + i + 1;
        }
}

/* f64: the keys [lo, hi) of table row `row`, the sink in the maximum and the denominator. */
void ref_one(const Problem& p, const int* table, int lo, int hi, int r, int h, double* out) {
    std::vector<double> s;
    std::vector<const uint16_t*> rows;
    const uint16_t* qv = &p.q[((size_t)r * p.heads + h) * L];
    const uint16_t* rv = &p.qr[((size_t)r * p.heads + h) * R];
    for (int t = lo; t < hi; ++t) {
        const int pg = table[t / p.bs];
        if (pg < 0 || pg >= p.pages) continue;
        const uint16_t* k = &p.kv[((size_t)pg * p.bs + t % p.bs) * W];
        double d = 0;
        for (int c = 0; c < L; ++c) d += (double)fb(qv[c]) * fb(k[c]);
        for (int c = 0; c < R; ++c) d += (double)fb(rv[c]) * fb(k[L + c]);
        s.push_back(d * kScale);
        rows.push_back(k);
    }
    for (int c = 0; c < L; ++c) out[c] = 0;
    if (s.empty()) return;
    const double sink = p.sinks[h];
    double mx = sink;
    for (double v : s) mx = std::max(mx, v);
    double den = std::exp(sink - mx);
    for (size_t j = 0; j < s.size(); ++j) {
        const double w = std::exp(s[j] - mx);
        den += w;
        for (int c = 0; c < L; ++c) out[c] += w * fb(rows[j][c]);
    }
    for (int c = 0; c < L; ++c) out[c] /= den;
}

std::vector<double> ref_paged(const Problem& p) {
    const int M = p.rows(), S = (int)p.qlen.size();
    std::vector<double> want((size_t)M * p.heads * L);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < p.qlen[s]; ++i) {
            const int r = p.cu[s] + i;
            const long pos = (long)p.used[s] - p.qlen[s] + i;
            const int hi = (int)std::min<long>(std::max<long>(pos + 1, 0), (long)p.blocks * p.bs);
            const int lo = p.window > 0 ? (int)std::max<long>(0, pos + 1 - p.window) : 0;
            for (int h = 0; h < p.heads; ++h)
                ref_one(p, &p.bt[(size_t)s * p.blocks], lo, hi, r, h,
                        &want[((size_t)r * p.heads + h) * L]);
        }
    return want;
}

std::vector<double> ref_sel(const Problem& p) {
    const int M = p.rows();
    std::vector<double> want((size_t)M * p.heads * L);
    for (int r = 0; r < M; ++r) {
        const int hi = std::clamp(p.sel_len[r], 0, p.sel_blocks * p.bs);
        for (int h = 0; h < p.heads; ++h)
            ref_one(p, &p.sel_bt[(size_t)r * p.sel_blocks], 0, hi, r, h,
                    &want[((size_t)r * p.heads + h) * L]);
    }
    return want;
}

struct Params {
    std::vector<RadParam> v;
    void i(const char* k, long long x) {
        RadParam p{};
        p.key = k;
        p.kind = RAD_P_INT;
        p.ival = x;
        v.push_back(p);
    }
    void f(const char* k, double x) {
        RadParam p{};
        p.key = k;
        p.kind = RAD_P_F64;
        p.dval = x;
        v.push_back(p);
    }
};

/* `stable` -1 leaves the parameter out (its default, 0). `scratch_cut` shortens the scratch;
 * `scr` (if given) gets the scratch's words after the call: the split form writes its slots' parts
 * there, the fused form only the accumulator of a row and head (`kAccWords` floats). */
int launch(const Problem& p, bool sel, int form, std::vector<uint16_t>& out, int stable = -1,
           int64_t scratch_cut = 0, std::vector<uint32_t>* scr = nullptr) {
    const int M = p.rows();
    out.assign((size_t)M * p.heads * L, 0xffff);
    auto& P = const_cast<Problem&>(p);
    std::vector<RadTensor> t;
    t.push_back(tens(P.q.data(), RAD_BF16, { M, p.heads, L }));
    t.push_back(tens(P.qr.data(), RAD_BF16, { M, p.heads, R }));
    t.push_back(tens(P.kv.data(), RAD_BF16, { p.pages, 1, p.bs, W }));
    Params ps;
    ps.i("M", M);
    ps.i("heads", p.heads);
    ps.i("latent_dim", L);
    ps.i("rope_dim", R);
    if (sel) {
        t.push_back(tens(P.sel_bt.data(), RAD_I32, { M, p.sel_blocks }));
        t.push_back(tens(P.sel_len.data(), RAD_I32, { M }));
        ps.i("blocks", p.sel_blocks);
    } else {
        t.push_back(tens(P.bt.data(), RAD_I32, { (int64_t)p.qlen.size(), p.blocks }));
        t.push_back(tens(P.used.data(), RAD_I32, { (int64_t)p.used.size() }));
        t.push_back(tens(P.cu.data(), RAD_I32, { (int64_t)p.cu.size() }));
        ps.i("window", p.window);
    }
    ps.i("block_size", p.bs);
    ps.f("scale", kScale);
    if (stable >= 0) ps.i("stable", stable);
    t.push_back(tens(P.sinks.data(), RAD_F32, { p.heads }));
    t.push_back(tens(out.data(), RAD_BF16, { M, p.heads, L }));
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = ps.v.data();
    a.n_p = (int)ps.v.size();
    const int64_t need =
        sel ? r4d_rad_mla_attn_sel_sink_scratch(&a) : r4d_rad_mla_attn_paged_sink_scratch(&a);
    std::vector<char> scratch((size_t)need + 16);
    a.scratch = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);
    a.scratch_bytes = need > scratch_cut ? need - scratch_cut : 0;
    r4d_mla_sink_pin_form(form);
    const int rc = sel ? r4d_rad_mla_attn_sel_sink(&a, nullptr)
                       : r4d_rad_mla_attn_paged_sink(&a, nullptr);
    r4d_mla_sink_pin_form(0);
    if (scr) {
        const uint32_t* w = (const uint32_t*)a.scratch;
        scr->assign(w, w + a.scratch_bytes / 4);
    }
    return rc;
}

/* The scratch hook's answer for a shape: it reads shapes and parameters only, so the operands need
 * no data behind them. */
int64_t hook_size(const Problem& p, bool sel, int stable) {
    const int M = p.rows();
    std::vector<RadTensor> t;
    t.push_back(tens(nullptr, RAD_BF16, { M, p.heads, L }));
    t.push_back(tens(nullptr, RAD_BF16, { M, p.heads, R }));
    t.push_back(tens(nullptr, RAD_BF16, { p.pages, 1, p.bs, W }));
    Params ps;
    ps.i("M", M);
    ps.i("heads", p.heads);
    ps.i("latent_dim", L);
    ps.i("rope_dim", R);
    if (sel) {
        t.push_back(tens(nullptr, RAD_I32, { M, p.sel_blocks }));
        t.push_back(tens(nullptr, RAD_I32, { M }));
        ps.i("blocks", p.sel_blocks);
    } else {
        t.push_back(tens(nullptr, RAD_I32, { (int64_t)p.qlen.size(), p.blocks }));
        t.push_back(tens(nullptr, RAD_I32, { (int64_t)p.used.size() }));
        t.push_back(tens(nullptr, RAD_I32, { (int64_t)p.cu.size() }));
        ps.i("window", p.window);
    }
    ps.i("block_size", p.bs);
    ps.f("scale", kScale);
    ps.i("stable", stable);
    t.push_back(tens(nullptr, RAD_F32, { p.heads }));
    t.push_back(tens(nullptr, RAD_BF16, { M, p.heads, L }));
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = ps.v.data();
    a.n_p = (int)ps.v.size();
    return sel ? r4d_rad_mla_attn_sel_sink_scratch(&a) : r4d_rad_mla_attn_paged_sink_scratch(&a);
}

int g_fail = 0;
void report(const std::string& name, bool ok, const char* detail = "") {
    std::printf("%-64s %s %s\n", name.c_str(), ok ? "PASS" : "FAIL", detail);
    g_fail += !ok;
}

bool close_to(const std::vector<uint16_t>& got, const std::vector<double>& want, double& max_err) {
    long bad = 0;
    max_err = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float g = fb(got[i]);
        const double e = std::fabs(g - want[i]);
        if (!std::isfinite(g) || e > 1e-5 + 0.008 * std::fabs(want[i])) ++bad;
        if (std::isfinite(g)) max_err = std::max(max_err, e);
    }
    return bad == 0;
}

/* Every form against the reference; forms 0..3 (same split count) bit for bit. */
void check_forms(const std::string& name, const Problem& p, bool sel) {
    const std::vector<double> want = sel ? ref_sel(p) : ref_paged(p);
    std::vector<uint16_t> first;
    for (int form : { 0, 1, 2, 3, 4 }) {
        if (form == 2 && p.heads <= 16) continue;
        std::vector<uint16_t> out;
        const int rc = launch(p, sel, form, out);
        char buf[128];
        if (rc != RAD_OK) {
            std::snprintf(buf, sizeof buf, "rc %d", rc);
            report(name + " form " + std::to_string(form), false, buf);
            continue;
        }
        double err;
        bool ok = close_to(out, want, err);
        std::snprintf(buf, sizeof buf, "max_abs %.2e", err);
        std::string note = buf;
        if (form == 0) first = out;
        else if (form != 4 && out != first) {
            ok = false;
            note += " (bytes differ from the rule's form)";
        }
        report(name + (sel ? " sel" : " paged") + " form " + std::to_string(form), ok, note.c_str());
    }
}

/* Each verify row's bytes against a one-row launch at its position over the same table. */
void check_verify(const std::string& name, const Problem& p) {
    std::vector<uint16_t> all;
    if (launch(p, false, 0, all) != RAD_OK) {
        report(name + " verify", false, "launch");
        return;
    }
    bool same = true;
    const int S = (int)p.qlen.size();
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < p.qlen[s]; ++i) {
            Problem one = p;
            one.qlen = { 1 };
            one.ctx = { p.used[s] - p.qlen[s] + i + 1 };
            one.used = one.ctx;
            one.cu = { 0, 1 };
            one.bt.assign(p.bt.begin() + (size_t)s * p.blocks,
                          p.bt.begin() + (size_t)(s + 1) * p.blocks);
            const int r = p.cu[s] + i;
            one.q.assign(p.q.begin() + (size_t)r * p.heads * L,
                         p.q.begin() + (size_t)(r + 1) * p.heads * L);
            one.qr.assign(p.qr.begin() + (size_t)r * p.heads * R,
                          p.qr.begin() + (size_t)(r + 1) * p.heads * R);
            std::vector<uint16_t> o;
            if (launch(one, false, 0, o) != RAD_OK ||
                !std::equal(o.begin(), o.end(), all.begin() + (size_t)r * p.heads * L))
                same = false;
        }
    report(name + " verify rows == decode rows (bytes)", same);
}

void check_paged_vs_sel(const std::string& name, Problem p) {
    make_sel_from_paged(p);
    std::vector<uint16_t> a, b;
    const int ra = launch(p, false, 0, a), rb = launch(p, true, 0, b);
    report(name + " paged(window 0) == sel (bytes)", ra == RAD_OK && rb == RAD_OK && a == b);
    check_forms(name, p, true);
}

/* The window edge: row at position pos; the key at pos-window (outside) and pos-window+1 (inside)
 * get rotary parts aligned with every head's rotary query, the outside one more strongly, so
 * either mistake dominates the output. */
void check_edge(int heads, int window, int ctx, std::mt19937& rng) {
    Problem p;
    make_paged(p, { 1 }, { ctx }, heads, 4, window, false, rng);
    for (auto& v : p.qr) v = bf(0.5f);
    const int pos = ctx - 1;
    auto boost = [&](int t, float x) {
        const int pg = p.bt[t / p.bs];
        uint16_t* k = &p.kv[((size_t)pg * p.bs + t % p.bs) * W];
        for (int c = 0; c < R; ++c) k[L + c] = bf(x);
    };
    boost(pos - window, 5.f);     /* rotary score 0.5*5*64*scale = 10 */
    boost(pos - window + 1, 3.f); /* 6 */
    for (auto& s : p.sinks) s = 0.f;
    const std::vector<double> want = ref_paged(p);
    std::vector<uint16_t> out;
    const int rc = launch(p, false, 0, out);
    double err;
    const bool ok = rc == RAD_OK && close_to(out, want, err);
    char buf[64];
    std::snprintf(buf, sizeof buf, "max_abs %.2e", rc == RAD_OK ? err : -1.0);
    report("window edge H" + std::to_string(heads) + " ctx " + std::to_string(ctx), ok, buf);
}

/* One row r of a paged problem as its own one-row launch at its position over the same table. */
Problem one_row(const Problem& p, int s, int i) {
    Problem one = p;
    one.qlen = { 1 };
    one.ctx = { p.used[s] - p.qlen[s] + i + 1 };
    one.used = one.ctx;
    one.cu = { 0, 1 };
    one.bt.assign(p.bt.begin() + (size_t)s * p.blocks, p.bt.begin() + (size_t)(s + 1) * p.blocks);
    const int r = p.cu[s] + i;
    one.q.assign(p.q.begin() + (size_t)r * p.heads * L, p.q.begin() + (size_t)(r + 1) * p.heads * L);
    one.qr.assign(p.qr.begin() + (size_t)r * p.heads * R,
                  p.qr.begin() + (size_t)(r + 1) * p.heads * R);
    return one;
}

/* stable 1: every row of a launch against its one-row launch (paged and, at window 0, the
 * selection form over the same pages), the launch against the f64 reference, and the HG forms
 * (`forms_pins` false where a case is about the rows alone and the forms are pinned elsewhere; the
 * f64 reference and the row-per-row law always run). */
void check_stable(const std::string& name, const Problem& p, bool forms_pins = true) {
    const std::vector<double> want = ref_paged(p);
    std::vector<uint16_t> all;
    int rc = launch(p, false, 0, all, 1);
    double err = 0;
    const bool ok = rc == RAD_OK && close_to(all, want, err);
    char buf[96];
    std::snprintf(buf, sizeof buf, "max_abs %.2e", err);
    report(name + " stable 1 vs f64", ok, buf);
    bool same = rc == RAD_OK;
    const int S = (int)p.qlen.size();
    for (int s = 0; same && s < S; ++s)
        for (int i = 0; same && i < p.qlen[s]; ++i) {
            std::vector<uint16_t> o;
            const int r = p.cu[s] + i;
            same = launch(one_row(p, s, i), false, 0, o, 1) == RAD_OK &&
                   std::equal(o.begin(), o.end(), all.begin() + (size_t)r * p.heads * L);
        }
    report(name + " stable 1: every row == its one-row launch (bytes)", same);
    if (forms_pins) {
        bool forms = true;
        for (int form : { 1, 2, 3, 5, 6 }) {
            if ((form == 2 && p.heads <= 16) || (form == 5 && p.heads <= 32)) continue;
            std::vector<uint16_t> o;
            forms = forms && launch(p, false, form, o, 1) == RAD_OK && o == all;
        }
        report(name + " stable 1: HG 1 / 2 / 4, the element load and the fused form equal (bytes)",
               forms);
    }
    if (p.window == 0) {
        Problem q = p;
        make_sel_from_paged(q);
        std::vector<uint16_t> b;
        report(name + " stable 1: paged(window 0) == sel (bytes)",
               launch(q, true, 0, b, 1) == RAD_OK && b == all);
    }
}

/* K-16: `stable 1` splits to the verify heights (48 rows, max_seqs x (max_spec + 1) = 8 x 6), where
 * a workgroup takes a slot rather than walking the row; `stable 0` keeps its measured 16 / 32 rows.
 * What the cap moves is the plan, not the bytes (the law is the same either way, and check_stable
 * pins it): the scratch hook sizes the partials at those rows, and a launch up to the cap leaves
 * them past the fused form's accumulator, while past the cap one workgroup walks the row. */
void check_verify_cap(std::mt19937& rng) {
    {
        /* The hook covers 48 rows x the law's slots (8 at a window of 512, 20 at window 0, 16 in the
         * selection form over blocks x block_size 2048) x heads x the part. */
        Problem p;
        make_paged(p, { 6, 6, 6, 6, 6, 6, 6, 6 }, { 600, 530, 511, 512, 513, 600, 700, 300 }, 32, 4,
                   512, false, rng);
        const int64_t split8 = 48 * 8 * 32 * kAccWords * 4;
        report("stable 1: the scratch covers the split form at 48 rows (8 slots)",
               hook_size(p, false, 1) >= split8);
        report("stable 0: the scratch is still 16 rows at the window",
               hook_size(p, false, 0) == 16 * 8 * 32 * kAccWords * 4);
        p.heads = 16;
        p.window = 0;
        const int64_t split20 = 48 * 20 * 16 * kAccWords * 4;
        report("stable 1: the scratch covers 20 slots at 48 rows (window 0)",
               hook_size(p, false, 1) >= split20);
        report("stable 0: the scratch is still 32 rows at window 0",
               hook_size(p, false, 0) == 32 * 20 * 16 * kAccWords * 4);
        p.sel_blocks = 512; /* blocks x block_size 2048: the law's 16 slots */
        const int64_t split16 = 48 * 16 * 16 * kAccWords * 4;
        report("stable 1: the scratch covers the selection form at 48 rows",
               hook_size(p, true, 1) >= split16);
        report("stable 0: the scratch is still 32 rows in the selection form",
               hook_size(p, true, 0) == 32 * 16 * 16 * kAccWords * 4);
    }
    /* The form at the cap, both of them: 48 rows split (a slot a workgroup, its part in the scratch
     * past the fused form's accumulator a row and head), 49 rows fuse. */
    for (int window : { 512, 0 })
        for (int rows : { 48, 49 }) {
            Problem p;
            make_paged(p, { rows }, { 600 }, 8, 4, window, true, rng);
            std::vector<uint16_t> o;
            std::vector<uint32_t> scr;
            const int rc = launch(p, false, 0, o, 1, 0, &scr);
            bool parts = false;
            for (size_t i = (size_t)rows * p.heads * kAccWords; i < scr.size(); ++i)
                parts = parts || scr[i] != 0;
            report("stable 1 w" + std::to_string(window) + ": " + std::to_string(rows) +
                       " rows run the " + (rows <= 48 ? "split" : "fused") + " form",
                   rc == RAD_OK && parts == (rows <= 48));
            if (window == 0 && rows == 48) {
                /* the selection form at the verify height, over the same pages: the whole split */
                Problem q = p;
                make_sel_from_paged(q);
                std::vector<uint16_t> b;
                report("stable 1 w0: 48 rows, paged == the selection form (bytes)",
                       launch(q, true, 0, b, 1) == RAD_OK && b == o);
            }
        }
}

void stable_suite(std::mt19937& rng, bool quick) {
    /* (a) launches of 2..64 rows of 1, 3 and 8 sequences: the row's bytes equal its M = 1 launch,
     * whatever M, the number of streams or the other rows. */
    for (int window : { 512, 0 })
        for (int M : { 2, 6, 16, 17, 33, 64 })
            for (int seqs : { 1, 3, 8 }) {
                if (seqs > M || (quick && M > 17)) continue;
                std::vector<int> qlen(seqs, M / seqs), ctx(seqs);
                for (int s = 0; s < M % seqs; ++s) ++qlen[s];
                for (int s = 0; s < seqs; ++s) ctx[s] = 520 + 37 * s + qlen[s];
                Problem p;
                make_paged(p, qlen, ctx, 8, 4, window, true, rng);
                check_stable("stable H8 w" + std::to_string(window) + " M " + std::to_string(M) +
                                 " seqs " + std::to_string(seqs),
                             p);
            }
    /* (c) lengths at the edges of chunks, super-chunks, the window and the law's last slot (20 x
     * 128 = 2560 at window 0), each a sequence of one row in one launch. */
    for (int window : { 512, 0 }) {
        std::vector<int> lens = { 0, 1, 15, 16, 17, 511, 512, 513, 2048, 2049 };
        if (window == 0) lens.push_back(2700);
        Problem p;
        make_paged(p, std::vector<int>(lens.size(), 1), lens, 8, 16, window, false, rng);
        check_stable("stable H8 w" + std::to_string(window) + " lengths 0..2049", p);
    }
    /* heads 16 / 32 / 64 (HG 1, 2, 4 by the rule and pinned) at a verify-sized launch */
    for (int heads : { 16, 32, 64 }) {
        if (quick && heads > 16) continue;
        Problem p;
        make_paged(p, { 6, 6, 6 }, { 600, 530, 1100 }, heads, 4, heads == 32 ? 0 : 512, true, rng);
        check_stable("stable H" + std::to_string(heads) + " 3 streams x 6", p);
    }
    /* gpu-mini's geometry: a verify of 8 rows of one sequence past the window's edge, the table as
     * wide as the context bound (3072 tokens), 64 heads at block 16 (sliding) and 32 at block 4
     * (dense sparse layer), stable 0 and 1 */
    for (int sliding : { 1, 0 }) {
        Problem p;
        const int bs = sliding ? 16 : 4;
        make_paged(p, { 8 }, { 554 }, sliding ? 64 : 32, bs, sliding ? 512 : 0, false, rng);
        const int wide = 3072 / bs;
        std::vector<int> t((size_t)wide, -1);
        std::copy(p.bt.begin(), p.bt.end(), t.begin());
        p.bt = t;
        p.blocks = wide;
        check_stable(std::string("gpu-mini ") + (sliding ? "sliding" : "dense") + " verify 8 at 554", p);
        check_verify(std::string("gpu-mini ") + (sliding ? "sliding" : "dense") + " stable 0 verify 8 at 554", p);
    }
    /* stable 0 equals stable 1 below its row cap; the table's width (the slots a launch needs)
     * moves no bit; a short scratch is refused at stable 1 and unsplit at stable 0. */
    {
        Problem p;
        make_paged(p, { 3, 2 }, { 700, 90 }, 16, 4, 512, true, rng);
        std::vector<uint16_t> a, b, c;
        launch(p, false, 0, a, 1);
        report("stable 0 == stable 1 below the row cap (bytes)",
               launch(p, false, 0, b, 0) == RAD_OK && launch(p, false, 0, c, -1) == RAD_OK &&
                   a == b && a == c);
        for (int window : { 512, 0 }) {
            p.window = window;
            launch(p, false, 0, a, 1);
            Problem wide = p;
            const int extra = 700;
            wide.bt.assign((size_t)2 * (p.blocks + extra), -1);
            for (int s = 0; s < 2; ++s)
                std::copy(p.bt.begin() + (size_t)s * p.blocks, p.bt.begin() + (size_t)(s + 1) * p.blocks,
                          wide.bt.begin() + (size_t)s * (p.blocks + extra));
            wide.blocks = p.blocks + extra;
            report("stable 1 w" + std::to_string(window) + ": table width moves no bit",
                   launch(wide, false, 0, b, 1) == RAD_OK && a == b);
        }
        p.window = 512; /* the hook's count is the launch's: a paged window 0 sizes for 20 slots */
        /* the split form's partials do not fit: the fused form, the same bytes */
        std::vector<uint16_t> want;
        launch(p, false, 0, want, 1);
        report("stable 1: no room for the partials runs the fused form (bytes)",
               launch(p, false, 0, a, 1, 16) == RAD_OK && a == want);
        /* no room for the accumulator either (5 rows x 16 heads x 516 floats): refused */
        const int64_t keep = (int64_t)5 * 16 * 516 * 4 - 16;
        std::vector<uint16_t> tmp;
        {
            /* the hook's size minus everything but `keep` bytes */
            Problem q = p;
            int64_t need = 0;
            {
                RadArgs args{};
                std::vector<RadParam> ps(4);
                ps[0].key = "M"; ps[0].kind = RAD_P_INT; ps[0].ival = q.rows();
                ps[1].key = "heads"; ps[1].kind = RAD_P_INT; ps[1].ival = q.heads;
                ps[2].key = "window"; ps[2].kind = RAD_P_INT; ps[2].ival = q.window;
                ps[3].key = "stable"; ps[3].kind = RAD_P_INT; ps[3].ival = 1;
                args.p = ps.data();
                args.n_p = 4;
                need = r4d_rad_mla_attn_paged_sink_scratch(&args);
            }
            report("stable 1: no room for the accumulator refused (RAD_E_SCRATCH)",
                   launch(q, false, 0, tmp, 1, need - keep) == RAD_E_SCRATCH);
        }
        report("stable 0: short scratch runs unsplit", launch(p, false, 0, a, 0, 16) == RAD_OK);
        report("stable 2 refused", launch(p, false, 0, a, 2) == RAD_E_SHAPE);
    }
    /* stable 0 past the row cap runs unsplit: within the bound, and still the reference */
    {
        Problem p;
        make_paged(p, { 20 }, { 600 }, 8, 4, 512, true, rng);
        std::vector<uint16_t> o;
        double err;
        const bool ok = launch(p, false, 0, o, 0) == RAD_OK && close_to(o, ref_paged(p), err);
        report("stable 0 past 16 rows (unsplit) vs f64", ok);
    }
    check_verify_cap(rng);
    /* The verify height itself, where the split form now runs: 8 streams x 6 rows (48) on the
     * sliding layer's shape at TP2 (32 heads, window 512). The HG forms are pinned at other row
     * counts and head counts, so this case is about the rows: f64 and the row-per-row law. */
    if (!quick) {
        Problem p;
        make_paged(p, { 6, 6, 6, 6, 6, 6, 6, 6 }, { 600, 530, 511, 512, 513, 1100, 2048, 700 }, 32,
                   4, 512, true, rng);
        check_stable("stable H32 w512 verify 8 x 6 (48 rows)", p, false);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(20261121);
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    /* heads x (sequences: query lengths, contexts) x window */
    struct C {
        int heads;
        std::vector<int> qlen, ctx;
        int window, bs;
        bool holes;
    };
    std::vector<C> cases = {
        { 8, { 1 }, { 1 }, 512, 4, false },
        { 8, { 1, 1, 1 }, { 15, 16, 17 }, 512, 4, true },
        { 16, { 2, 1 }, { 511, 513 }, 512, 4, true },
        { 32, { 4 }, { 600 }, 512, 16, true },
        { 64, { 1, 3 }, { 512, 520 }, 512, 4, true },
        { 64, { 2 }, { 300 }, 0, 4, true },
        { 32, { 1, 2 }, { 33, 530 }, 0, 4, true },
    };
    if (quick) cases.resize(3);
    for (const auto& c : cases) {
        Problem p;
        make_paged(p, c.qlen, c.ctx, c.heads, c.bs, c.window, c.holes, rng);
        std::string name = "H" + std::to_string(c.heads) + " w" + std::to_string(c.window) +
                           " bs" + std::to_string(c.bs) + " ctx";
        for (int x : c.ctx) name += " " + std::to_string(x);
        check_forms(name, p, false);
        check_verify(name, p);
        if (c.window == 0) check_paged_vs_sel(name, p);
    }
    /* A verify of 8 rows past the window, and the same rows at window 0 against the selection. */
    {
        Problem p;
        make_paged(p, { 8 }, { 530 }, 16, 4, 512, true, rng);
        check_verify("H16 w512 verify 8 at 530", p);
        p.window = 0;
        check_verify("H16 w0 verify 8 at 530", p);
        check_paged_vs_sel("H16 w0 verify 8 at 530", p);
    }
    stable_suite(rng, quick);
    check_edge(16, 512, 600, rng);
    if (!quick) check_edge(64, 512, 530, rng);
    /* The selection form on its own tables: scattered pages, holes, a zero length. */
    {
        Problem p;
        make_paged(p, { 3 }, { 200 }, 32, 4, 0, false, rng);
        make_sel_from_paged(p);
        for (int r = 0; r < 3; ++r)
            for (int b = 0; b < p.sel_blocks; ++b)
                p.sel_bt[(size_t)r * p.sel_blocks + b] =
                    b % 13 == 5 ? -1 : b % 11 == 7 ? p.pages + 1 : (b * 7 + r * 3) % p.pages;
        p.sel_len = { 0, 77, 200 };
        check_forms("sel scattered", p, true);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
