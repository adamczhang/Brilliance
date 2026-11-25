/* hyv4flash_engram.h -- the Engram memory in front of a layer (HYV4FlashEngramLayer,
 * kernels/engram.md).
 *
 *   ids  = engram_ids(tokens)          [T, heads] hashed bi- and trigram ids, the history window a
 *                                      CONV state of the layer's own (one issue a layer a step)
 *   e    = embed_lookup_q(ids, table)  [T, heads * hd]: E4M3 codes and a bf16 scale a row (the
 *                                      bf16 recipe), or the bf16 table as shipped
 *   e    = the E4M3 round trip of each row, for a bf16 table under engram_fp8_embedding (the model
 *          was trained to read its rows so): quant_act_fp8 with one group a row, dequant
 *   e    = engram_norm(e)              one rms a head
 *   key  = engram_key(e) + bias        [T, hc * n], the hc per-stream key projections joined
 *   val  = engram_value(e) + bias      [T, n]
 *   h   += engram_gate(key, val, h)    per stream, in place; masked rows (pad, image, video) untouched
 *
 * The norms' eps is HF's `nn.RMSNorm(dim)` default, finfo(dtype).eps (DECISIONS U1):
 * RADIANCE_HY4_ENGRAM_EPS=f32 (1.19e-7, the default: HF in f32, vLLM, llama.cpp and the friend's
 * reference `ref/reference.py`), bf16 (0.0078125: HF run in bf16), or a number. */
#ifndef HYV4FLASH_ENGRAM_H
#define HYV4FLASH_ENGRAM_H

#include "hyv4flash_common.h"

namespace hyv4flash {

/* RADIANCE_HY4_ENGRAM_AHEAD: the read-ahead switch for the first prefill pass. 0 hashes no
 * ahead ids; any other value hashes them at every placement (the pre-read-ahead-gating A/B
 * arm); unset hashes them only where a gather reads them (Engram::step). */
inline bool no_ahead() {
    const char* e = std::getenv("RADIANCE_HY4_ENGRAM_AHEAD");
    return e && *e && std::strcmp(e, "0") == 0;
}

/* Nonzero for any value but 0: hash the ahead ids even where no gather reads them. */
inline bool ahead_forced() {
    const char* e = std::getenv("RADIANCE_HY4_ENGRAM_AHEAD");
    return e && *e && std::strcmp(e, "0") != 0;
}

inline double engram_eps() {
    const char* e = std::getenv("RADIANCE_HY4_ENGRAM_EPS");
    if (!e || !*e || !std::strcmp(e, "f32")) return 1.1920928955078125e-07;
    if (!std::strcmp(e, "bf16")) return 0.0078125;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (end != e && !*end && v > 0) return v;
    fprintf(stderr, "radiance: hyv4flash: RADIANCE_HY4_ENGRAM_EPS='%s' is none of bf16, f32 or a "
                    "positive number; using f32 (1.19e-7)\n", e);
    return 1.1920928955078125e-07;
}

/* The planes every Engram layer shares.
 *
 * The ids and the gathered rows are host-domain buffers, as rad_block_ple.h declares its own: the
 * table's placement (--ngram-placement) is decided after the declare, and a table on `disk` is
 * gathered by libavx's host kernel, which reads the ids and writes the rows in the host arena. The
 * host arena is mapped, so a device gather (vram, ram) and the norm behind it read and write them
 * over the link instead (3 KiB of rows and 64 bytes of ids a token).
 *
 * The read-ahead (RadBatch::n_ahead, docs/OPS.md "ahead_ids and ahead"): a prefill chunk's
 * engram_ids also hashes up to kAheadChunks steps of the prompt that follows it into `ahead`, and
 * the gather hands those ids to libavx's reader thread, which fills its row cache while the chunk
 * runs. With the table on a drive a fresh prompt's chunks otherwise wait on their row reads. No
 * output depends on them; a device gather ignores them, so step() hashes them only when the
 * gather dispatches to the host domain (rad_op_domain) -- disk placement, or a locked RAM copy
 * a card cannot read zero-copy. */
struct EngramWire {
    /* The ahead queue's depth in steps: core/sched/batch.cpp's k_ahead_chunks (as kAheadChunks in
     * rad_block_ple.h); a larger offer is clamped to the buffer. */
    static constexpr int64_t kAheadChunks = 2;
    rad_buf ids = 0;      /* [T, heads] i32, host domain */
    rad_buf ahead = 0;    /* [kAheadChunks * T, heads] i32, host domain: the following prompt's ids */
    rad_buf mask = 0;     /* [T] i32 */
    rad_buf raw = 0;      /* [T, heads * hd] the gathered rows, host domain */
    rad_buf q = 0;        /* [T * heads, hd] e4m3, the round trip's codes */
    rad_buf qs = 0;       /* [T * heads, 1] f32 */
    rad_buf rt = 0;       /* [T, heads * hd] the round trip */
    rad_buf e = 0;        /* [T, heads * hd] normed */
    rad_buf key = 0;      /* [T, hc * n] */
    rad_buf val = 0;      /* [T, n] */
    rad_buf before = 0;   /* [T, hc * n] the streams before the add (the engram_out tap) */
    rad_buf m1 = 0;       /* [T, hc * n] -1, the tap's subtraction */
    int declare(RadBuilder* b, Names& nm, const Shape& s, const Geom& g) {
        const int64_t T = g.max_tok, H = s.engram_heads, HD = s.engram_head_dim, W = H * HD;
        ids = decl_b(b, nm.f("eg.ids"), RAD_I32, { T, H }, RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
        ahead = decl_b(b, nm.f("eg.ids_ahead"), RAD_I32, { kAheadChunks * T, H }, RAD_BUF_TRANSIENT,
                       RAD_DOMAIN_HOST);
        mask = decl_b(b, nm.f("eg.mask"), RAD_I32, { T });
        raw = decl_b(b, nm.f("eg.rows"), RAD_BF16, { T, W }, RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
        q = decl_b(b, nm.f("eg.rows.q"), RAD_F8E4M3, { T * H, HD });
        qs = decl_b(b, nm.f("eg.rows.s"), RAD_F32, { T * H, 1 });
        rt = decl_b(b, nm.f("eg.rows.rt"), RAD_BF16, { T, W });
        e = decl_b(b, nm.f("eg.normed"), RAD_BF16, { T, W });
        key = decl_b(b, nm.f("eg.key"), RAD_BF16, { T, s.hc * s.n_embd });
        val = decl_b(b, nm.f("eg.value"), RAD_BF16, { T, s.n_embd });
        if (taps_on()) {
            /* PERSIST: a tap's scratch is not in the declared op order's liveness. */
            before = decl_b(b, nm.f("eg.before"), RAD_BF16, { T, s.hc * s.n_embd }, RAD_BUF_PERSIST);
            m1 = decl_b(b, nm.f("eg.minus_one"), RAD_BF16, { T, s.hc * s.n_embd }, RAD_BUF_PERSIST);
        }
        return ids && ahead && mask && raw && q && qs && rt && e && key && val &&
                       (!taps_on() || (before && m1))
                   ? RAD_OK
                   : RAD_E_INVAL;
    }
};

struct Engram {
    Shape s{};
    Geom g{};
    EngramWire w{};
    int layer = 0;
    rad_buf h = 0;
    rad_kvgroup hist = 0;
    bool table_bf16 = true, roundtrip = false;
    bool hashless = false;   /* the container lacks the hash constants (an optional --add) */
    rad_weight table = 0, table_scale = 0, map = 0, mult = 0, vsz = 0, off = 0, head_gain = 0;
    rad_weight key_w = 0, key_b = 0, key_g = 0, query_g = 0, val_w = 0, val_b = 0, val_g = 0,
               branch = 0;
    rad_op op_ids = 0, op_ids_dec = 0, op_gather = 0, op_quant = 0, op_dequant = 0, op_norm = 0;
    rad_op op_key = 0, op_val = 0, op_gate = 0;
    rad_op op_tap_copy = 0, op_tap_m1 = 0, op_tap_mul = 0, op_tap_add = 0;
    Tap t_out{}, t_ids{};

    /* Joins `n` per-stream (or per-head) checkpoint tensors `<prefix><i><suffix>` along dim 0,
     * eight a map: successive CONCAT maps for one name extend it in call order. */
    static int map_join(RadBuilder* b, Names& nm, const char* declared, const char* prefix,
                        const char* suffix, int64_t n) {
        for (int64_t i = 0; i < n; i += 8) {
            RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
            for (int64_t k = i; k < n && k < i + 8; ++k)
                m.src[m.n_src++] = nm.ckpt("%s%lld%s", prefix, (long long)k, suffix);
            RAD_ARCH_TRY(rad_decl_name_map(b, &m));
        }
        return RAD_OK;
    }

    /* `rows` is the layer's table height (the sum of its head moduli). */
    int declare(RadBuilder* b, Names& nm, const Shape& shape, const Geom& geom, int l, int64_t rows,
                const EngramWire& wire, rad_buf streams, rad_kvgroup history) {
        s = shape;
        g = geom;
        w = wire;
        layer = l;
        h = streams;
        hist = history;
        const int64_t T = g.max_tok, E = s.n_embd, HD = s.engram_head_dim, H = s.engram_heads,
                      W = H * HD;
        const auto grp = grp_layer(l);
        const char* ep = nm.ckpt("model.layers.%d.engram_layer", l);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, hist));

        /* The table: E4M3 rows and a bf16 or f32 scale a row (a recipe), or bf16 as shipped. A vocabulary
         * table, though not the token one: a few rows a token, far too large for a card. Model-level
         * (grp_model), as rad_block_ple.h declares its tables: the planner takes an embed_lookup_q table
         * off the cards (--ngram-placement) only when it is a model-level weight. */
        const char* tn = nm.f("blk.%d.engram_table.weight", l);
        RAD_ARCH_TRY(map_copy(b, tn, nm.ckpt("%s.multi_head_embedding.embedding.weight", ep)));
        RadEncoding te{};
        const bool known = weight_enc(b, tn, &te);
        table_bf16 = !known || (rad_enc_is(&te, "plain") && te.n_planes == 1 &&
                                te.plane[0].dtype == RAD_BF16);
        if (table_bf16) {
            table = decl_w(b, tn, RAD_BF16, { rows, HD }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        } else {
            const RadEncPlane* sc = rad_enc_plane(&te, "scale");
            if (!(te.n_planes == 2 && te.plane[0].dtype == RAD_F8E4M3 && sc &&
                  (sc->dtype == RAD_BF16 || sc->dtype == RAD_F32) && sc->block[0] == 1)) {
                char en[256];
                rad_enc_format(&te, en, sizeof en);
                fprintf(stderr, "radiance: hyv4flash: '%s' is %s; the Engram table is served "
                                "bf16 or as E4M3 rows under a bf16 or f32 scale a row "
                                "(fp8_e4m3*bf16[1x*], data/recipes/hy4-flash-bf16.recipe; "
                                "fp8_e4m3*f32[1x*], hy4-flash-bf16-f32.recipe)\n",
                        tn, en);
                return RAD_E_UNSUPPORTED;
            }
            table = decl_view(b, tn, nullptr, "codes", RAD_F8E4M3, { rows, HD }, RAD_ACCESS_VOCAB,
                              RAD_SHARD_NONE, grp_model());
            table_scale = decl_view(b, nm.f("blk.%d.engram_table.scale", l), tn, "scale",
                                    sc->dtype, { rows, 1 }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
            if (!table_scale) return RAD_E_INVAL;
        }
        if (!table) return RAD_E_INVAL;
        /* The bf16 table is what the model reads only after each row's E4M3 round trip
         * (`engram_fp8_embedding`); the E4M3 table already is that (within one code step: HANDOFF). */
        roundtrip = table_bf16 && s.engram_fp8;

        const char* mn = nm.f("blk.%d.engram_token_map", l);
        RAD_ARCH_TRY(map_copy(b, mn, nm.ckpt("%s.lookup_table", ep)));
        /* The token map's length is the tokenizer's, below the embedding's padded vocabulary. */
        {
            RadEncoding e{};
            int64_t shp[RAD_MAX_RANK] = {};
            uint32_t rank = 0;
            if (rad_weight_encoding(b, mn, &e, shp, &rank) == RAD_OK && rank >= 1)
                s.engram_map = shp[0];
        }
        if (s.engram_map < 1 || s.engram_pad < 0 || s.engram_pad >= s.engram_map)
            return refuse("the Engram token map is empty or does not cover engram_pad_id");
        map = decl_w(b, mn, RAD_I32, { s.engram_map }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        /* The hash constants are not in the checkpoint: data/hy4flash_engram_tables.py writes them
         * for rad-convert --add. Without them the model cannot hash, so they are required. */
        const char* mu = nm.f("blk.%d.engram_mult", l);
        const char* vs = nm.f("blk.%d.engram_vocab_sizes", l);
        const char* of = nm.f("blk.%d.engram_offsets", l);
        RAD_ARCH_TRY(map_copy(b, mu, nm.ckpt("engram.%d.hash_multipliers", l)));
        RAD_ARCH_TRY(map_copy(b, vs, nm.ckpt("engram.%d.head_vocab_sizes", l)));
        RAD_ARCH_TRY(map_copy(b, of, nm.ckpt("engram.%d.head_offsets", l)));
        /* Optional, so a checkpoint converts without them (and the convert test checks they are left
         * out); a step without them cannot hash and fails by name. */
        RadEncoding me{};
        hashless = weight_enc(b, tn, &me) && !weight_enc(b, mu, &me);
        mult = decl_w(b, mu, RAD_I64, { s.engram_ngram }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp,
                      1);
        vsz = decl_w(b, vs, RAD_I64, { H }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp, 1);
        off = decl_w(b, of, RAD_I64, { H }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp, 1);
        if (!map || !mult || !vsz || !off) return RAD_E_INVAL;
        if (hashless)
            rad_note(b, "hyv4flash: layer %d's Engram hash constants are not in the checkpoint; "
                        "write them with data/hy4flash_engram_tables.py <checkpoint> "
                        "tables.safetensors and convert with rad-convert --add "
                        "tables.safetensors, or the model cannot serve", l);

        const char* hn = nm.f("blk.%d.engram_head_norm.weight", l);
        RAD_ARCH_TRY(map_join(b, nm, hn, nm.ckpt("%s.lookup_per_head_norms.", ep), ".weight", H));
        head_gain = decl_w(b, hn, RAD_F32, { W }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        const char* kn = nm.f("blk.%d.engram_key.weight", l);
        const char* kb = nm.f("blk.%d.engram_key.bias", l);
        const char* kg = nm.f("blk.%d.engram_key_norm.weight", l);
        const char* qg = nm.f("blk.%d.engram_query_norm.weight", l);
        const char* bs = nm.f("blk.%d.engram_branch_scale.weight", l);
        RAD_ARCH_TRY(map_join(b, nm, kn, nm.ckpt("%s.key_projs.", ep), ".weight", s.hc));
        RAD_ARCH_TRY(map_join(b, nm, kb, nm.ckpt("%s.key_projs.", ep), ".bias", s.hc));
        RAD_ARCH_TRY(map_join(b, nm, kg, nm.ckpt("%s.norm_key.", ep), ".weight", s.hc));
        RAD_ARCH_TRY(map_join(b, nm, qg, nm.ckpt("%s.norm_query.", ep), ".weight", s.hc));
        RAD_ARCH_TRY(map_join(b, nm, bs, nm.ckpt("%s.unified_per_hc_scales.", ep), "", s.hc));
        const char* vn = nm.f("blk.%d.engram_value.weight", l);
        RAD_ARCH_TRY(map_copy(b, vn, nm.ckpt("%s.value_proj.weight", ep)));
        for (const char* n : { kn, vn })
            if (!is_plain_bf16(b, n))
                return refuse("a quantised Engram projection is not served yet (WP10)");
        key_w = decl_w(b, kn, RAD_BF16, { s.hc * E, W }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        key_b = decl_w(b, kb, RAD_F32, { s.hc * E }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        key_g = decl_w(b, kg, RAD_F32, { s.hc * E }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        query_g = decl_w(b, qg, RAD_F32, { s.hc * E }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        branch = decl_w(b, bs, RAD_F32, { s.hc * E }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        val_w = decl_w(b, vn, RAD_BF16, { E, W }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        val_b = decl_f32(b, nm.f("blk.%d.engram_value.bias", l), nm.ckpt("%s.value_proj.bias", ep),
                         { E }, grp);
        val_g = decl_f32(b, nm.f("blk.%d.engram_value_gain.weight", l),
                         nm.ckpt("%s.unified_pre_gamma", ep), { E }, grp);
        if (!head_gain || !key_w || !key_b || !key_g || !query_g || !branch || !val_w || !val_b ||
            !val_g)
            return RAD_E_INVAL;

        const double eps = engram_eps();
        for (int dec = 0; dec < 2; ++dec) {
            const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("heads", H),
                                    RAD_INT("ngram", s.engram_ngram), RAD_INT("pad", s.engram_pad),
                                    RAD_INT("map_size", s.engram_map),
                                    RAD_INT("image", s.image_token),
                                    RAD_INT("video", s.video_token) };
            const rad_weight ws[] = { map, mult, vsz, off };
            rad_op& op = dec ? op_ids_dec : op_ids;
            /* The prefill form also writes the following prompt's ids (`ahead`). */
            op = dec ? rw(b, rad_decl_op(b, "engram_ids", ps, 7, ws, 4), {}, { w.ids, w.mask })
                     : rw(b, rad_decl_op(b, "engram_ids", ps, 7, ws, 4), {},
                          { w.ids, w.mask, w.ahead });
            HY4_NEED(op, "engram_ids");
        }
        {
            const rad_weight ws[2] = { table, table_scale };
            const RadParam ps[] = { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", HD),
                                    RAD_INT("n_vocab", rows), RAD_STR("dtype", "bf16") };
            op_gather = rw(b, rad_decl_op(b, "embed_lookup_q", ps, 4, ws, table_bf16 ? 1 : 2),
                           { w.ids, w.ahead }, { w.raw });
            HY4_NEED(op_gather, "embed_lookup_q");
        }
        if (roundtrip) {
            op_quant = mk(b, "quant_act_fp8",
                          { RAD_RANGE("M", 1, T * H), RAD_INT("n", HD), RAD_INT("group", HD),
                            RAD_STR("dtype", "bf16") },
                          {}, { w.raw }, { w.q, w.qs });
            op_dequant = mk(b, "dequant",
                            { RAD_RANGE("M", 1, T * H), RAD_INT("n", HD), RAD_INT("group", HD),
                              RAD_STR("dtype", "bf16") },
                            {}, { w.q, w.qs }, { w.rt });
            HY4_NEED(op_quant && op_dequant, "the Engram rows' E4M3 round trip (quant_act_fp8, "
                                             "dequant)");
        }
        const rad_buf normed_in = roundtrip ? w.rt : w.raw;
        op_norm = mk(b, "engram_norm",
                     { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("n", HD),
                       RAD_F64("eps", eps), RAD_STR("dtype", "bf16") },
                     { head_gain }, { normed_in }, { w.e });
        op_key = mk(b, "gemm_nt_bias",
                    { RAD_RANGE("M", 1, T), RAD_INT("N", s.hc * E), RAD_INT("K", W),
                      RAD_STR("dtype", "bf16") },
                    { key_w, key_b }, { w.e }, { w.key });
        op_val = mk(b, "gemm_nt_bias",
                    { RAD_RANGE("M", 1, T), RAD_INT("N", E), RAD_INT("K", W),
                      RAD_STR("dtype", "bf16") },
                    { val_w, val_b }, { w.e }, { w.val });
        op_gate = mk(b, "engram_gate",
                     { RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_INT("hc", s.hc),
                       RAD_F64("eps", eps), RAD_STR("dtype", "bf16") },
                     { key_g, query_g, val_g, branch }, { w.key, w.val, w.mask, h }, { h });
        HY4_NEED(op_norm, "engram_norm");
        HY4_NEED(op_key && op_val, "gemm_nt_bias");
        HY4_NEED(op_gate, "engram_gate");

        /* engram_out: the residual the add made, the streams after it less a copy before it. */
        if (taps_on()) {
            RAD_ARCH_TRY(t_out.declare(b, nm, T, "engram_out", l, RAD_BF16, s.hc * E, w.before));
            const int64_t HW = s.hc * E;
            op_tap_copy = decl_copy(b, T, HW, RAD_BF16, h, w.before);
            op_tap_m1 = mk(b, "fill",
                           { RAD_RANGE("M", 1, T), RAD_INT("n", HW), RAD_STR("dtype", "bf16"),
                             RAD_F64("value", -1.0) },
                           {}, {}, { w.m1 });
            op_tap_mul = mk(b, "mul",
                            { RAD_RANGE("M", 1, T), RAD_INT("n", HW), RAD_STR("dtype", "bf16") },
                            {}, { w.before, w.m1 }, { w.before });
            op_tap_add = mk(b, "add",
                            { RAD_RANGE("M", 1, T), RAD_INT("n", HW), RAD_STR("dtype", "bf16") },
                            {}, { h, w.before }, { t_out.buf });
            HY4_NEED(op_tap_copy && op_tap_m1 && op_tap_mul && op_tap_add,
                     "the engram_out tap (cast, fill, mul, add)");
            RAD_ARCH_TRY(t_ids.declare(b, nm, T, "engram_hash_ids", l, RAD_I32, H, w.ids));

        }
        return RAD_OK;
    }

    /* The step's ids, the decode half with num_accepted (its window read where the last verify
     * left it), the prefill half without (each chunk's tail lands at offset zero): ple_conv's two
     * layouts, as rad_block_ple.h issues ngram_ids. */
    bool step(RadCtx* c, const RadBatch* batch) const {
        const int64_t T = batch->n_tok, S = batch->n_seq, H = s.engram_heads;
        const RadKVGroupBatch* hb = kv_batch(batch, hist);
        const int32_t* sidx = hb ? hb->state_index : nullptr;
        const int64_t pitch = hb && hb->state_index_pitch > 0 ? hb->state_index_pitch : 1;
        if (!sidx) {
            rad_step_fail(c, "hyv4flash: the Engram history's state slots are missing");
            return false;
        }
        if (hashless) {
            rad_step_fail(c, "hyv4flash: the container has no Engram hash constants "
                             "(blk.<l>.engram_mult, engram_vocab_sizes, engram_offsets); convert "
                             "with rad-convert --add the output of data/hy4flash_engram_tables.py");
            return false;
        }
        int64_t D = 0, DT = 0;
        batch_split(batch, &D, &DT);
        const int64_t P = S - D;
        const RadOperand tok = praw(batch->token_ids, RAD_I32, T);
        const RadOperand pos = praw(batch->positions, RAD_I32, T);
        /* cache_idx: column 0 of each sequence's state slots. */
        if (D > 0) {
            RadOperand sd = praw2(sidx, RAD_I32, D, pitch);
            sd.cols = 1;
            RAD_ISSUE_N(c, op_ids_dec, T, tok, kv_cache(hist, layer),
                        praw(batch->cu_seqlens, RAD_I32, D + 1), pos, RAD_W(map), RAD_W(mult),
                        RAD_W(vsz), RAD_W(off), praw(batch->num_accepted, RAD_I32, D), sd,
                        brows(w.ids, T), brows(w.mask, T));
        }
        /* The prompt after the last prefill chunk (RadBatch::n_ahead, stored in token_ids past the
         * step's), hashed by the same launch as a continuation of that sequence. Only the host
         * gather's reader thread reads those ids, so the hashing runs only when the gather goes to
         * the host domain (the table on disk, or a RAM copy the card cannot read zero-copy); a
         * device gather ignores them. ahead_forced() hashes them anyway (the A/B arm). */
        const int64_t A =
            P > 0 && !no_ahead() &&
                    (ahead_forced() || rad_op_domain(c, op_gather) == RAD_DOMAIN_HOST)
                ? std::min<int64_t>(batch->n_ahead, EngramWire::kAheadChunks * g.max_tok)
                : 0;
        if (P > 0) {
            RadOperand sp = praw2(sidx + D * pitch, RAD_I32, P, pitch);
            sp.cols = 1;
            RAD_ISSUE_N(c, op_ids, T, praw(batch->token_ids, RAD_I32, T + A), kv_cache(hist, layer),
                        praw(batch->cu_seqlens + D, RAD_I32, P + 1), pos, RAD_W(map), RAD_W(mult),
                        RAD_W(vsz), RAD_W(off), RAD_NONE, sp, brows(w.ids, T), brows(w.mask, T),
                        A > 0 ? brows(w.ahead, A) : RAD_NONE);
        }
        t_ids.step(c, brows(w.ids, T), T);
        RAD_ISSUE_N(c, op_gather, T * H, brows(w.ids, T), RAD_W(table),
                    table_bf16 ? RAD_NONE : RAD_W(table_scale), brows(w.raw, T),
                    A > 0 ? brows(w.ahead, A) : RAD_NONE);
        if (roundtrip) {
            RadOperand raw = brows(w.raw, T);
            RAD_ISSUE_N(c, op_quant, T * H, raw, brows(w.q, T * H), brows(w.qs, T * H));
            RAD_ISSUE_N(c, op_dequant, T * H, brows(w.q, T * H), brows(w.qs, T * H),
                        brows(w.rt, T));
        }
        RAD_ISSUE_N(c, op_norm, T, brows(roundtrip ? w.rt : w.raw, T), RAD_W(head_gain),
                    brows(w.e, T));
        RAD_ISSUE_N(c, op_key, T, brows(w.e, T), RAD_W(key_w), RAD_W(key_b), RAD_NONE,
                    brows(w.key, T));
        RAD_ISSUE_N(c, op_val, T, brows(w.e, T), RAD_W(val_w), RAD_W(val_b), RAD_NONE,
                    brows(w.val, T));
        if (op_tap_copy) RAD_ISSUE_N(c, op_tap_copy, T, brows(h, T), brows(w.before, T));
        RAD_ISSUE_N(c, op_gate, T, brows(w.key, T), brows(w.val, T), brows(w.mask, T), RAD_W(key_g),
                    RAD_W(query_g), RAD_W(val_g), RAD_W(branch), brows(h, T));
        if (op_tap_add) {
            RAD_ISSUE_N(c, op_tap_m1, T, brows(w.m1, T));
            RAD_ISSUE_N(c, op_tap_mul, T, brows(w.before, T), brows(w.m1, T), brows(w.before, T));
            RAD_ISSUE_N(c, op_tap_add, T, brows(h, T), brows(w.before, T), brows(t_out.buf, T));
        }
        return true;
    }
};

}  // namespace hyv4flash
#endif
