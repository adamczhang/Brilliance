/* ar_bytes_test.cpp -- the all-reduce byte counter (Ctx::ar_bytes) counts the ops that fold an
 * all-reduce in, not only the conventional all_reduce.
 *
 * Brilliance (ar-bytes-count). Upstream bills only `all_reduce`, whose issued `n` is the message's
 * element count. A TP2 prefill step's residual writes run as ar_hc_write and ar_gather_hc_write,
 * whose `n` is a row count; they moved about 2.8 GiB a rank a step on Flash-Next and /stats showed
 * none of it. The Program is built by hand around one recording kernel per op, as issue_div_test
 * builds its own, and runs on whichever device backend is linked.
 */
#include "rad_test.h"

#include "device/device.h"
#include "runtime/ctx.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using rad::Ctx;
using rad::CtxDesc;
using rad::Program;

namespace {

int g_launches = 0;
int k_record(const RadArgs*, RadStream) { ++g_launches; return RAD_OK; }

/* Schemas with the shape of libr4d's (r4d_rows.cpp): the all_reduce's message is operand 0 and its
 * ranged parameter is the element count; ar_hc_write's message is operand 0 and its ranged
 * parameter the row count M; ar_gather_hc_write's message is operand 5, after the gather's. */
const RadParamSpec   g_p_ar[]  = { { "numel", RAD_P_INT, RAD_REQUIRED, 0 } };
const RadOperandSpec g_o_ar[]  = { { "x", RAD_OPD_INOUT, 0 } };
const RadParamSpec   g_p_hc[]  = { { "M", RAD_P_INT, RAD_REQUIRED, 0 },
                                   { "n", RAD_P_INT, RAD_REQUIRED, 0 } };
const RadOperandSpec g_o_hc[]  = { { "y", RAD_OPD_INOUT, 0 }, { "inj", RAD_OPD_IN, 0 },
                                   { "h", RAD_OPD_INOUT, 0 } };
const RadOperandSpec g_o_ghc[] = { { "y_expert", RAD_OPD_IN, 0 }, { "expert_w", RAD_OPD_IN, 0 },
                                   { "sorted_tok", RAD_OPD_IN, 0 }, { "shared", RAD_OPD_IN, 1 },
                                   { "shared_gate", RAD_OPD_IN, 1 }, { "y", RAD_OPD_INOUT, 0 },
                                   { "inj", RAD_OPD_IN, 0 }, { "h", RAD_OPD_INOUT, 0 } };
const RadParamSpec   g_p_m[]   = { { "M", RAD_P_INT, RAD_REQUIRED, 0 } };
const RadOperandSpec g_o_xy[]  = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };

const RadOpSchema g_s_ar   = { "all_reduce", g_p_ar, 1, g_o_ar, 1, "test" };
const RadOpSchema g_s_hc   = { "ar_hc_write", g_p_hc, 2, g_o_hc, 3, "test" };
const RadOpSchema g_s_ghc  = { "ar_gather_hc_write", g_p_hc, 2, g_o_ghc, 8, "test" };
const RadOpSchema g_s_norm = { "ar_rmsnorm_quant_fp8", g_p_hc, 2, g_o_xy, 2, "test" };
const RadOpSchema g_s_add  = { "add", g_p_m, 1, g_o_xy, 2, "test" };

std::function<void(Ctx&)> g_body;
void trampoline(RadCtx* c, const RadBatch*) { if (g_body) g_body(*static_cast<Ctx*>(c)); }

constexpr int64_t kRows  = 64;     /* every buffer is kRows x kWidth */
constexpr int64_t kWidth = 32;     /* the row length, the fused ops' `n` */

struct Fixture {
    Program       P;
    std::vector<RadKernelInfo>  infos;
    std::vector<rad::KernelRow> rows;
    Ctx           ctx;
    RadBatch      batch{};
    size_t        used = 0;                   /* rows handed out */
    rad_buf       bf = 0, f32 = 0, i32 = 0;   /* a bf16 buffer, an f32 one, an i32 one */
    rad_op        ar = 0, hc = 0, ghc = 0, norm = 0, add = 0, hc_no_n = 0;

    rad_buf buffer(const char* name, uint32_t dt) {
        rad::BufferInfo b;
        b.name = name;
        b.decl.dtype = dt;
        b.decl.rank = 2;
        b.decl.shape[0] = kRows;
        b.decl.shape[1] = kWidth;
        b.decl.domain = RAD_DOMAIN_DEVICE;
        b.bytes = rad_dtype_bytes(dt, kRows * kWidth);
        b.arena_offset = (int64_t)(P.buffers.size() - 1) * 16384;
        P.buffers.push_back(b);
        return (rad_buf)(P.buffers.size() - 1);
    }

    /* One op, one device band up to `hi` on its ranged key, a row of its own that records. */
    rad_op op(const char* name, const RadOpSchema* s, const char* ranged, int64_t hi, bool with_n) {
        const size_t k = used;
        rad::OpInfo o;
        o.op = name;
        o.index = (int32_t)P.ops.size();
        o.ranged_key = ranged;
        o.range_lo = 1;
        o.range_hi = hi;
        o.schema = s;
        if (with_n) o.base.set_i("n", kWidth);
        rad::Band b;
        b.hi = hi;
        b.miss[RAD_DOMAIN_HOST] = "no host kernel";
        rad::Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
        infos[k].op = name;
        r.row = &rows[k];
        r.geom.set_i(ranged, hi);
        if (with_n) r.geom.set_i("n", kWidth);
        o.bands.push_back(b);
        P.ops.push_back(o);
        ++used;
        return (rad_op)(P.ops.size() - 1);
    }

    Fixture() {
        P.meta.n_layers = 1;
        P.arena_bytes = 4 * 16384;
        P.scratch_bytes = 0;
        P.weights.resize(1);
        P.buffers.resize(1);
        P.ops.resize(1);
        bf  = buffer("bf", RAD_BF16);
        f32 = buffer("f32", RAD_F32);
        i32 = buffer("i32", RAD_I32);
        /* Rows are pointed at by the bands, so both vectors are sized before any op is made. */
        infos.resize(6);
        rows.resize(6);
        for (size_t i = 0; i < infos.size(); ++i) {
            infos[i].name = "t_record";
            infos[i].op = "t";
            infos[i].family = "test";
            infos[i].domain = RAD_DOMAIN_DEVICE;
            infos[i].launch = k_record;
            rows[i].info = &infos[i];
            rows[i].plugin = "fake";
        }
        ar          = op("all_reduce", &g_s_ar, "numel", kRows * kWidth, false);
        hc          = op("ar_hc_write", &g_s_hc, "M", kRows, true);
        ghc         = op("ar_gather_hc_write", &g_s_ghc, "M", kRows, true);
        norm        = op("ar_rmsnorm_quant_fp8", &g_s_norm, "M", kRows, true);
        add         = op("add", &g_s_add, "M", kRows, false);
        hc_no_n     = op("ar_hc_write", &g_s_hc, "M", kRows, false);
    }
    int start() {
        CtxDesc d;
        d.program = &P;
        d.rank = 0;
        d.world_size = 1;
        d.arch_step = trampoline;
        return ctx.init(d);
    }
    int step(const std::function<void(Ctx&)>& body) {
        batch.phase = RAD_PHASE_DECODE;
        batch.n_tok = (int32_t)kRows;
        batch.n_seq = 1;
        g_body = body;
        const int s = ctx.run_step(&batch);
        g_body = nullptr;
        return s;
    }
};

const rad::OpInfo& info(Fixture& f, rad_op h) { return f.P.ops[(size_t)h]; }

}  /* namespace */

/* The rule bind applies, op by op: what is billed, which operand is the message, and how many
 * elements one unit of the issued `n` is. */
TEST(ar_billing_names_the_message_and_its_width) {
    Fixture f;
    int32_t opd = -1;
    int64_t w = -1;
    CHECK(rad::ar_billing(info(f, f.ar), &opd, &w));
    CHECK_EQ(opd, 0);
    CHECK_EQ(w, 1);
    CHECK(rad::ar_billing(info(f, f.hc), &opd, &w));
    CHECK_EQ(opd, 0);
    CHECK_EQ(w, kWidth);
    CHECK(rad::ar_billing(info(f, f.ghc), &opd, &w));
    CHECK_EQ(opd, 5);
    CHECK_EQ(w, kWidth);
    CHECK(rad::ar_billing(info(f, f.norm), &opd, &w));
    CHECK_EQ(opd, 0);
    CHECK_EQ(w, kWidth);
    /* Not an all-reduce; a fused op whose row length or row range is missing is not billed
     * rather than billed at a guessed width. */
    CHECK(!rad::ar_billing(info(f, f.add), &opd, &w));
    CHECK(!rad::ar_billing(info(f, f.hc_no_n), &opd, &w));
    rad::OpInfo unranged = info(f, f.hc);
    unranged.ranged_key.clear();
    CHECK(!rad::ar_billing(unranged, &opd, &w));
    /* Every fused family libr4d declares is on the list. */
    for (const char* n : { "ar_rmsnorm_post_quant_fp8", "ar_rmsnorm_had_quant_i8g",
                           "ar_rmsnorm_quant_i8", "ar_rmsnorm_quant_i8g", "ar_ln_had_quant_i8" }) {
        rad::OpInfo o = info(f, f.hc);
        o.op = n;
        CHECK(rad::ar_billing(o, &opd, &w));
        CHECK_EQ(opd, 0);
    }
}

/* Through the issue path: each billed issue adds a call and its message's bytes, rows times the
 * row length times the message operand's element size, and nothing else does. */
TEST(ar_bytes_counts_the_fused_writes) {
    Fixture f;
    REQUIRE_EQ(f.start(), RAD_OK);
    g_launches = 0;
    const int64_t c0 = f.ctx.ar_calls(), y0 = f.ctx.ar_bytes();
    CHECK_EQ(c0, 0);
    CHECK_EQ(y0, 0);
    CHECK_OK(f.step([&](Ctx& c) {
        /* The conventional all_reduce: 1024 bf16 elements, 2048 bytes (upstream's count). */
        RadOperand a[1] = { RAD_B(f.bf) };
        CHECK_OK(c.issue(f.ar, a, 1, 1024));
    }));
    CHECK_EQ(f.ctx.ar_calls(), 1);
    CHECK_EQ(f.ctx.ar_bytes(), 2048);

    CHECK_OK(f.step([&](Ctx& c) {
        /* ar_hc_write over 48 rows of 32 bf16: 3072 bytes. */
        RadOperand h[3] = { RAD_B(f.bf), RAD_B(f.f32), RAD_B(f.f32) };
        CHECK_OK(c.issue(f.hc, h, 3, 48));
        /* ar_gather_hc_write over 10 rows: the message is operand 5 (bf16, 640 bytes), not the
         * f32 expert outputs at operand 0, which would bill 1280. The shared pair is absent. */
        RadOperand g[8] = { RAD_B(f.f32), RAD_B(f.f32), RAD_B(f.i32), RAD_NONE, RAD_NONE,
                            RAD_B(f.bf), RAD_B(f.f32), RAD_B(f.f32) };
        CHECK_OK(c.issue(f.ghc, g, 8, 10));
        /* A fused norm over 64 rows: 4096 bytes. */
        RadOperand q[2] = { RAD_B(f.bf), RAD_B(f.bf) };
        CHECK_OK(c.issue(f.norm, q, 2, 64));
        /* Not an all-reduce, and a fused op bind could not size: no bytes, no calls. */
        RadOperand x[2] = { RAD_B(f.bf), RAD_B(f.bf) };
        CHECK_OK(c.issue(f.add, x, 2, 64));
        CHECK_OK(c.issue(f.hc_no_n, h, 3, 64));
    }));
    CHECK_EQ(g_launches, 6);
    CHECK_EQ(f.ctx.ar_calls(), 1 + 3);
    CHECK_EQ(f.ctx.ar_bytes(), 2048 + 48 * kWidth * 2 + 10 * kWidth * 2 + 64 * kWidth * 2);
}

RAD_TEST_MAIN()
