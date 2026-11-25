/* hyv4flash_calib.h -- Hy4-flash's calibration taps (PLAN §4.1).
 *
 * Two taps record a run (`radiance --calib-corpus FILE --calib-dir DIR`, docs/ARCHITECTURES.md):
 *
 *   CalibFull (arch/common/rad_calib_full.h) the whole Hessian of every trunk linear's input, one
 *     safetensors file a distinct input, `<DIR>/<site>.safetensors`. The site is named after the
 *     first weight that reads it and its metadata `linears` lists every declared weight stem that
 *     does (`blk.0.attn_q_a,blk.0.attn_kv_a,blk.0.attn_gate`). libquant does NOT read these files:
 *     a recipe's `gptq calib=DIR` looks for `gram.r<rank>.<the weight's own stem>.bin`
 *     (RADGRAM2, libquant/lq_hess.cpp), so `tools/calib_st2gram.py` turns each site file into one
 *     such file for every stem in its `linears`.
 *   MoeFP8's own tap (rad_block_moe_fp8.h) the latent experts: the pooled gate_up Gram
 *     `gram.r0.blk.L.ffn_gate_up_exps.bin` and the pooled and per-expert down Grams
 *     `gram.r0.blk.L.ffn_down_exps[.E].bin`, RADGRAM2 as written, read by libquant as they are. It
 *     is driven by the environment (RADIANCE_CALIB_DIR / _TOKENS / _LAYERS, rad_arch.h);
 *     `map_cli_to_env` sets those from the CLI values before the first block reads them.
 *
 * The sites of a layer l (K the input's width) and the weights that read them:
 *
 *   blk.l.attn_q_a.input        K hidden   xa, the attention norm's output
 *                                          attn_q_a attn_kv_a attn_gate (sparse: index_key index_head)
 *   blk.l.attn_q_b.input        K q_lora   the normed q_a                  attn_q_nope attn_q_rope
 *                                          (sparse: index_query_nope index_query_rope)
 *   blk.l.attn_kv_b.input       K kv_lora  the normed latent (bf16 weight, recorded all the same)
 *   blk.l.attn_output.input     K H*v      the gated attention output     attn_output
 *   blk.0.ffn_gate_up.input     K hidden   layer 0's MLP, then blk.0.ffn_down.input (K 13824)
 *   blk.l.ffn_latent_in.input   K hidden   ffn_latent_in ffn_gate_up_shexp
 *   blk.l.ffn_down_shexp.input  K shared   ffn_down_shexp
 *   blk.l.ffn_latent_out.input  K latent   the routed sum in the latent   ffn_latent_out
 *   blk.l.engram_key.input      K 1536     the normed Engram rows         engram_key engram_value
 *   output.input                K hidden   the final norm's output         output
 *
 * The router, norms, iHC parameters, sinks and the pooling projections are not linears a recipe
 * quantises and have no site. A site reads a buffer right after the op that fills it, before the
 * next layer reuses the shared plane; the hooks are in hyv4flash.cpp's step.
 *
 * TP: the attention output's input (heads a rank), the dense MLP's down input and the shared
 * expert's are split by columns: those sites declare k_local = K / world and CalibFull gathers them
 * to the whole row; one rank (layer % world) records each layer. The routed experts are whole a
 * rank under calibration (Ffn::place) and each rank writes its own experts' files. */
#ifndef HYV4FLASH_CALIB_H
#define HYV4FLASH_CALIB_H

#include <arch/rad_calib_full.h>

#include <mutex>

#include "hyv4flash_attn.h"
#include "hyv4flash_engram.h"
#include "hyv4flash_indexer.h"
#include "hyv4flash_moe.h"

namespace hyv4flash {

/* The CLI's --calib-dir / --calib-tokens / --calib-layers for the blocks that read the
 * environment (MoeFP8's tap). A value the caller already exported must agree: a run recording
 * the trunk into one directory and the experts into another would leave the Grams a recipe needs
 * in two places. Returns RAD_OK when nothing is being recorded. Call before the first block's
 * probe (rad_arch.h's calib_dir() caches the first read). */
inline int map_cli_to_env_once(const RadBuildCtx* ctx);
inline int map_cli_to_env(const RadBuildCtx* ctx) {
    /* Every rank's thread declares; the first sets the environment and the others wait for it, so
     * no rank reads RADIANCE_CALIB_* (rad_arch.h caches the first read) before it is set. */
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    return map_cli_to_env_once(ctx);
}
inline int map_cli_to_env_once(const RadBuildCtx* ctx) {
    if (!ctx->calib_dir || !*ctx->calib_dir) return RAD_OK;
    const char* have = std::getenv("RADIANCE_CALIB_DIR");
    if (have && *have && std::strcmp(have, ctx->calib_dir) != 0) {
        fprintf(stderr, "radiance: hyv4flash: --calib-dir %s and RADIANCE_CALIB_DIR=%s name two "
                        "directories; the trunk and the experts record to one. Unset the variable "
                        "(the plugin sets it from --calib-dir)\n", ctx->calib_dir, have);
        return RAD_E_INVAL;
    }
    setenv("RADIANCE_CALIB_DIR", ctx->calib_dir, 0);
    if (ctx->calib_tokens > 0) {
        const std::string t = std::to_string((long long)ctx->calib_tokens);
        const char* e = std::getenv("RADIANCE_CALIB_TOKENS");
        if (e && *e && std::atoll(e) != (long long)ctx->calib_tokens)
            fprintf(stderr, "radiance: hyv4flash: RADIANCE_CALIB_TOKENS=%s but the corpus has %s "
                            "tokens; the experts' Grams are written at the former\n", e, t.c_str());
        setenv("RADIANCE_CALIB_TOKENS", t.c_str(), 0);
    }
    if (ctx->calib_layer_hi > 0 && !std::getenv("RADIANCE_CALIB_LAYERS")) {
        const std::string r = std::to_string((int)ctx->calib_layer_lo) + "-" +
                              std::to_string((int)ctx->calib_layer_hi - 1);
        setenv("RADIANCE_CALIB_LAYERS", r.c_str(), 0);
    }
    return RAD_OK;
}

struct Calib {
    /* One layer's site indices into cf.sites, -1 where this run records none. */
    struct Layer {
        int xa = -1, qr = -1, kvb = -1, ao = -1, xf = -1, down = -1, sdown = -1, lout = -1,
            eng = -1;
    };
    CalibFull cf;
    std::vector<Layer> layers;
    int head = -1;
    int64_t lat_w = 0;   /* kv_lora */
    /* At more ranks each holds the routed sum of its own experts only (MoeFP8 no_reduce, the
     * reduction runs behind latent_out): latent_out's whole input is the sum over ranks, so a
     * copy of the plane is reduced here (exact) and tapped. */
    rad_buf lat_src = 0, lat_sum = 0;
    rad_op op_lat_copy = 0, op_lat_ar = 0;

    bool on() const { return cf.on; }

    /* From the build context; nothing unless --calib-dir. The widest input of any site sizes the
     * scratch and the widest split one (the attention output, the down projections) the gather. */
    int begin(RadBuilder* b, Names& nm, const Geom& g, const RadBuildCtx* ctx, const Shape& s,
              rad_buf lat_x) {
        layers.assign((size_t)s.n_layer, Layer{});
        lat_w = s.kv_lora;
        int64_t split = std::max<int64_t>({ s.sparse.v_width(), s.sliding.v_width(), s.n_ff,
                                            s.n_ff_shared });
        int64_t wide = std::max<int64_t>({ s.n_embd, s.q_lora, s.kv_lora, split, s.latent,
                                           s.engram_heads * s.engram_head_dim });
        RAD_ARCH_TRY(cf.begin(b, nm, g, ctx, s.n_layer, split, wide));
        if (cf.on && g.world > 1 && lat_x) {
            lat_src = lat_x;
            lat_sum = decl_b(b, nm.f("calib.latent_sum"), RAD_BF16, { g.max_tok, s.latent });
            op_lat_copy = decl_copy(b, g.max_tok, s.latent, RAD_BF16, lat_src, lat_sum);
            int rc = RAD_OK;
            op_lat_ar = decl_all_reduce(b, g, s.latent, lat_sum, &rc);
            if (!lat_sum || !op_lat_copy || rc != RAD_OK) return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* The sites of layer l; `x` is the block-input plane the two norms write. The indexer is
     * null on a sliding layer. */
    int layer(RadBuilder* b, const Geom& g, const Shape& s, int l, rad_buf x, const Attn& at,
              bool sparse, const Ffn& ffn, const Engram* eg) {
        if (!cf.records(l)) return RAD_OK;
        Layer& y = layers[(size_t)l];
        const std::string p = "blk." + std::to_string(l) + ".";
        /* `k` is the whole input's width, `k_local` this rank's columns: the attention output, the
         * dense MLP's and the shared expert's down projections are split by input columns at TP
         * (CalibFull gathers them), every other input is whole on every rank. */
        auto add = [&](int* slot, const std::string& stem, const std::string& lin, rad_buf buf,
                       int64_t k, int64_t k_local = 0) {
            *slot = (int)cf.sites.size();
            return cf.site(b, g, l, stem + ".input", lin, buf, k_local ? k_local : k, k);
        };
        auto wn = [&](std::initializer_list<const char*> names) {
            std::string r;
            for (const char* n : names) r += (r.empty() ? "" : ",") + p + n;
            return r;
        };
        const AttnWire& w = at.w;
        std::string xa = wn({ "attn_q_a", "attn_kv_a", "attn_gate" });
        std::string qb = wn({ "attn_q_nope", "attn_q_rope" });
        if (sparse) {
            xa += "," + wn({ "index_key", "index_head" });
            qb += "," + wn({ "index_query_nope", "index_query_rope" });
        }
        RAD_ARCH_TRY(add(&y.xa, p + "attn_q_a", xa, x, s.n_embd));
        RAD_ARCH_TRY(add(&y.qr, p + "attn_q_b", qb, w.qa, s.q_lora));
        RAD_ARCH_TRY(add(&y.kvb, p + "attn_kv_b", p + "attn_kv_b", w.latc, s.kv_lora));
        RAD_ARCH_TRY(add(&y.ao, p + "attn_output", p + "attn_output", w.ag,
                         at.a.v_width() * g.world, at.a.v_width()));
        if (!ffn.moe_layer) {
            RAD_ARCH_TRY(add(&y.xf, p + "ffn_gate_up", p + "ffn_gate_up", x, s.n_embd));
            RAD_ARCH_TRY(add(&y.down, p + "ffn_down", p + "ffn_down", ffn.mlp.ff.x, s.n_ff, s.n_ff_r));
        } else {
            RAD_ARCH_TRY(add(&y.xf, p + "ffn_latent_in",
                             wn({ "ffn_latent_in", "ffn_gate_up_shexp" }), x, s.n_embd));
            RAD_ARCH_TRY(add(&y.sdown, p + "ffn_down_shexp", p + "ffn_down_shexp",
                             ffn.shared.ff.x, s.n_ff_shared, s.n_ff_shared_r));
            RAD_ARCH_TRY(add(&y.lout, p + "ffn_latent_out", p + "ffn_latent_out", lat_sum ? lat_sum : ffn.w.lat.x,
                             s.latent));
        }
        if (eg)
            RAD_ARCH_TRY(add(&y.eng, p + "engram_key", wn({ "engram_key", "engram_value" }),
                             eg->w.e, s.engram_heads * s.engram_head_dim));
        return RAD_OK;
    }

    /* The lm_head's input, recorded by the run that ends at the last layer. */
    int output(RadBuilder* b, const Geom& g, const Shape& s, rad_buf final_hidden) {
        if (!cf.on || cf.hi != s.n_layer) return RAD_OK;
        head = (int)cf.sites.size();
        return cf.site(b, g, -1, "output.input", "output", final_hidden, s.n_embd, s.n_embd);
    }

    void tap(RadCtx* c, int site, int64_t T) {
        if (site >= 0) cf.tap(c, (size_t)site, T);
    }

    /* After attn.front: the norm's output, the normed q_a, and the normed latent (copied out of
     * the cache row, whose rows hold the rotary key after it). */
    void after_front(RadCtx* c, int l, const Attn& at, int64_t T) {
        const Layer& y = layers[(size_t)l];
        if (y.xa < 0) return;
        tap(c, y.xa, T);
        tap(c, y.qr, T);
        const int64_t L = at.s.kv_lora, R = at.a.qk_rope;
        RAD_ISSUE_N(c, at.op_lat_copy, T, bcol_at(at.w.row, 0, L + R, 0, L, T),
                    brows(at.w.latc, T));
        tap(c, y.kvb, T);
    }
    void after_attn(RadCtx* c, int l, int64_t T) { tap(c, layers[(size_t)l].ao, T); }
    void after_ffn_norm(RadCtx* c, int l, int64_t T) { tap(c, layers[(size_t)l].xf, T); }
    void after_ffn(RadCtx* c, int l, int64_t T) {
        const Layer& y = layers[(size_t)l];
        tap(c, y.down, T);
        tap(c, y.sdown, T);
        if (y.lout >= 0 && lat_sum) {
            RAD_ISSUE_N(c, op_lat_copy, T, brows(lat_src, T), brows(lat_sum, T));
            issue_all_reduce(c, op_lat_ar, lat_sum, T, cf.sites[(size_t)y.lout].K);
        }
        tap(c, y.lout, T);
    }
    void after_engram(RadCtx* c, int l, int64_t T) { tap(c, layers[(size_t)l].eng, T); }
    void after_step(RadCtx* c, int64_t T) {
        tap(c, head, T);
        cf.after_step(c, T);
    }
};

}  // namespace hyv4flash
#endif
