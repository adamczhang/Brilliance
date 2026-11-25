/* hyv4flash_common.h -- what every block of the Hy4-flash graph shares: the geometry read from the
 * config, the op and weight helpers, the debug taps and the independent hyper-connection.
 *
 * Included by hyv4flash.cpp only (the blocks are header-only, in one translation unit). */
#ifndef HYV4FLASH_COMMON_H
#define HYV4FLASH_COMMON_H

#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace hyv4flash {
using namespace rad::arch;

/* One attention geometry: the sparse layers', the sliding layers', or the drafter's. */
struct AttnGeom {
    int64_t heads = 0, qk_nope = 0, qk_rope = 0, v = 0;
    int64_t q_width() const { return heads * (qk_nope + qk_rope); }
    int64_t kv_width() const { return heads * (qk_nope + v); }
    int64_t v_width() const { return heads * v; }
    int64_t qk() const { return qk_nope + qk_rope; }
};

struct Shape {
    int64_t n_layer = 0, n_embd = 0, n_vocab = 0, hc = 0;
    int64_t q_lora = 0, kv_lora = 0;
    AttnGeom sparse{}, sliding{};
    std::vector<int> is_sliding;   /* per layer */
    std::vector<int> is_moe;       /* per layer */
    /* feed-forward */
    int64_t n_ff = 0, n_expert = 0, n_ff_exp = 0, latent = 0, n_ff_shared = 0, top_k = 0;
    double route_scale = 1.0, swiglu_limit = 0.0;
    int norm_topk = 1;
    /* the compressed indexer: heads x dim, the pooling ratio, the groups a query recalls
     * (group_topk - 1 closed ones plus its own), and the context at or below which a sparse
     * layer attends densely (group_topk * ratio, HF's `candidate_topk`) */
    int64_t ix_heads = 0, ix_dim = 0, ix_pool = 0, ix_group_topk = 0, dense_limit = 0;
    /* RADIANCE_HY4_INDEXER=fp8 or int8: the group keys and queries on GLM's packed page, quantised
     * (fp8: E4M3 at a power-of-two scale, vLLM's IndexPool; int8: symmetric, f32 scale amax / 127)
     * instead of HF's bf16 keys and f32 scores (the default, `exact`). ix_fp8 is the packed wiring
     * and is set for both; ix_i8 selects the int8 codes. */
    bool ix_fp8 = false;
    bool ix_i8 = false;
    /* Engram: `layers` are layer indices (the config's ids are one above them) */
    std::vector<int> engram_layers;
    std::vector<int64_t> engram_rows;   /* each layer's table, the sum of its head moduli */
    int64_t engram_ngram = 0, engram_heads = 0, engram_head_dim = 0, engram_map = 0;
    int64_t engram_pad = 0, image_token = -1, video_token = -1;
    bool engram_fp8 = false;
    /* hyper-connections, rotary, window */
    double hc_eps = 1e-6, hc_mag = 2.0, eps = 1e-6;
    double theta_sparse = 1e7, theta_sliding = 1e4;
    int64_t window = 0;   /* keys a sliding row sees, its own token included */
    /* DFly2 and the tower */
    int64_t dfly_layers = 0;
    int64_t v_layers = 0, v_embd = 0, v_ff = 0, v_patch = 0, v_chan = 0, v_merge = 0, v_pos = 0;
    int64_t v_heads = 0, v_side = 0;   /* v_side: the position grid's side, max_image_size / patch */
    double v_eps = 1e-5;
    /* M-RoPE (PLAN D6): the layer kinds' `mrope_section` ("12 10 10"), and whether this program
     * rotates at the three-component positions (`mrope`: set when the tower is served, so every
     * rope reads RadBatch::rope_pos; a text-only program keeps gptj, the same rotation at equal
     * components). */
    std::string mrope_sparse, mrope_sliding;
    bool mrope = false;

    /* This rank's share (PLAN §1.5, set by split_ranks; at one rank the whole). The attention's
     * heads are whole and contiguous a rank (q_nope, q_rope, kv_b, attn_gate and the sinks by
     * rows, attn_output by columns); the dense MLP and the shared expert split their width evenly;
     * the routed experts are either whole a rank (`experts_whole`: experts [expert_base,
     * expert_base + n_expert_r), for bf16 experts and calibration) or a slice of every expert's
     * width (n_ff_exp_r, the quantised serving form). Everything else is replicated. The geometry
     * above stays the model's: the drafter and the tower are declared whole. */
    int world = 1, rank = 0;
    /* --calib-layers (RadBuildCtx::calib_layer_lo .. calib_layer_hi, hi exclusive; both 0: every layer). */
    int calib_lo = 0, calib_hi = 0;
    AttnGeom sparse_r{}, sliding_r{};
    int64_t n_ff_r = 0, n_ff_shared_r = 0, n_expert_r = 0, expert_base = 0, n_ff_exp_r = 0;
    bool experts_whole = true;
    const AttnGeom& local(bool sliding_layer) const { return sliding_layer ? sliding_r : sparse_r; }
};

inline int refuse(const char* what) {
    fprintf(stderr, "radiance: hyv4flash: %s\n", what);
    return RAD_E_UNSUPPORTED;
}

/* An op with its liveness, from lists: the RAD_OP macros take fixed arrays, and most ops here are
 * declared from a loop. Returns 0 where no library serves it. */
inline rad_op mk(RadBuilder* b, const char* name, std::initializer_list<RadParam> ps,
                 std::initializer_list<rad_weight> ws, std::initializer_list<rad_buf> reads,
                 std::initializer_list<rad_buf> writes) {
    const rad_op h = rad_decl_op(b, name, ps.begin(), (int)ps.size(),
                                 ws.size() ? ws.begin() : nullptr, (int)ws.size());
    return rw(b, h, reads, writes);
}

/* An op that must resolve: a graph with a hole would look servable. */
#define HY4_NEED(h, what)                                                                      \
    do {                                                                                       \
        if (!(h)) {                                                                            \
            fprintf(stderr, "radiance: hyv4flash: no loaded library serves %s\n", (what));     \
            return RAD_E_INVAL;                                                                \
        }                                                                                      \
    } while (0)

/* A weight the reference keeps in f32 (a norm's gain, a hyper-connection table, a sink, the
 * indexer's pooling, the router's correction): widened from bf16 at load where the checkpoint
 * stores it so. A recipe that quantised one is refused by name: nothing here reads its codes. */
inline rad_weight decl_f32(RadBuilder* b, const char* name, const char* src,
                           std::initializer_list<int64_t> shape, RadWeightGroup grp,
                           int access = RAD_ACCESS_PER_TOKEN, int shard = RAD_SHARD_NONE) {
    if (map_copy(b, name, src) != RAD_OK) return 0;
    RadEncoding e{};
    if (weight_enc(b, name, &e) && !(rad_enc_is(&e, "plain") && e.n_planes == 1)) {
        char en[256];
        rad_enc_format(&e, en, sizeof en);
        fprintf(stderr, "radiance: hyv4flash: '%s' is %s; it is a parameter the model applies "
                        "in f32, which this plugin reads only as stored (bf16 or f32). Convert it "
                        "without a recipe rule.\n", name, en);
        return 0;
    }
    return decl_w(b, name, RAD_F32, shape, access, shard, grp);
}

/* The all-reduce of a rank's partial [rows, n] in place, before a norm that needs the full sum:
 * exact bf16 (summed in f32 in rank order, rounded once), whatever wire the deployment allows,
 * because the norm behind it and the calibration read the sum. 0 at one rank. At more ranks a
 * library that serves no exact reduction is a refusal (PLAN §1.5), returned in `*rc`. */
inline rad_op decl_all_reduce(RadBuilder* b, const Geom& g, int64_t n, rad_buf x, int* rc,
                              int overlap = 0) {
    *rc = RAD_OK;
    if (g.world <= 1) return 0;
    /* `overlap` 1 declares the twin the second lane may issue: the library picks a launch narrow
     * enough that the GEMM beside it loses nothing (docs/TP4.md, the pipelined prefill). It is a
     * separate op because the row a step resolves to is the whole launch's shape. */
    const rad_op h = overlap
        ? mk(b, "all_reduce",
             { RAD_INT("world_size", g.world), RAD_RANGE("numel", n, g.max_tok * n),
               RAD_STR("dtype", "bf16"), RAD_INT("exact", 1),
               RAD_INT("min_bytes", g.wire_min_bytes), RAD_INT("overlap", overlap) },
             {}, { x }, { x })
        : mk(b, "all_reduce",
             { RAD_INT("world_size", g.world), RAD_RANGE("numel", n, g.max_tok * n),
               RAD_STR("dtype", "bf16"), RAD_INT("exact", 1),
               RAD_INT("min_bytes", g.wire_min_bytes) },
             {}, { x }, { x });
    if (!h) {
        fprintf(stderr, "radiance: hyv4flash: no loaded library serves an exact bf16 all_reduce "
                        "over %d ranks; the tensor-parallel graph needs one before each sandwich "
                        "norm\n", g.world);
        *rc = RAD_E_UNSUPPORTED;
    }
    return h;
}
inline void issue_all_reduce(RadCtx* c, rad_op op, rad_buf x, int64_t rows, int64_t n) {
    if (op && rows > 0) RAD_ISSUE_N(c, op, rows * n, brows(x, rows), RAD_NONE);
}

/* A plain bf16 weight that an op other than a linear reads (kv_b, the Engram table's bf16 form). */
inline bool is_plain_bf16(RadBuilder* b, const char* name) {
    RadEncoding e{};
    if (!weight_enc(b, name, &e)) return true;   /* no model behind the builder: as declared */
    return rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16;
}

/* `rmsnorm` over `n` columns with an f32 gain. */
inline rad_op decl_rmsnorm(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight gain,
                           rad_buf in, rad_buf out) {
    return mk(b, "rmsnorm",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                RAD_STR("dtype", "bf16") },
              { gain }, { in }, { out });
}

/* A copy, `cast` to the same dtype: views to and from strided column slices. */
inline const char* dt_name(uint32_t dt) {
    return dt == RAD_F32 ? "f32" : dt == RAD_I32 ? "i32" : "bf16";
}
inline rad_op decl_copy(RadBuilder* b, int64_t rows, int64_t n, uint32_t dt, rad_buf in,
                        rad_buf out) {
    return mk(b, "cast",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("from", dt_name(dt)),
                RAD_STR("to", dt_name(dt)) },
              {}, { in }, { out });
}

/* rope (interleaved pairs, `gptj`) over `heads` heads of `dim`, all of them rotated, in place.
 * The text trunk's M-RoPE with equal components is this rotation (plan D6). With `sections`
 * (a program that serves pictures, Shape::mrope) it is `imrope_gptj`, the reference's interleaved
 * M-RoPE over the same pairs, issued with the step's [3, T] rotary positions (rope_pos_of).
 *
 * `op` names the op to declare: "rope", or "rope_rows", the same op under its row-form name.
 * The row form is one thread per (token, pair) looping the heads -- one `sincosf` a (token, pair)
 * instead of one a (token, head, pair) -- and the kernel's launch takes it at 64+ rows and 4+ heads
 * (r4d_model_bf16.hip:1530-1541), so the same declaration keeps the per-(head, pair) launch at
 * decode and for a narrow rope. Both names are byte-identical, and every kernel library carries
 * both rows, so the name is the opt-in: "rope_rows" goes on a rope that rotates the query heads of
 * a whole prefill chunk, which is where the `sincosf` count is `heads` times what it needs to be. */
inline rad_op decl_rope_gptj(RadBuilder* b, int64_t rows, int64_t heads, int64_t dim,
                             double theta, rad_buf x, const char* sections = nullptr,
                             const char* op = "rope") {
    if (sections && *sections)
        return mk(b, op,
                  { RAD_RANGE("M", 1, rows), RAD_INT("head_dim", dim), RAD_INT("n_head", heads),
                    RAD_INT("n_head_kv", 0), RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                    RAD_STR("mode", "imrope_gptj"), RAD_INT("rotary_dim", dim),
                    RAD_STR("sections", sections) },
                  {}, { x }, { x });
    return mk(b, op,
              { RAD_RANGE("M", 1, rows), RAD_INT("head_dim", dim), RAD_INT("n_head", heads),
                RAD_INT("n_head_kv", 0), RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                RAD_STR("mode", "gptj"), RAD_INT("rotary_dim", dim) },
              {}, { x }, { x });
}
/* The positions a rope of this program reads: the step's [3, T] rotary planes under M-RoPE (the
 * core carries them whenever the program declares an encoder), else the token positions. */
inline RadOperand rope_pos_of(const Shape& s, const RadBatch* batch, int64_t T) {
    if (s.mrope && batch->rope_pos) return praw2(batch->rope_pos, RAD_I32, 3, T);
    return praw(batch->positions, RAD_I32, T);
}

/* ------------------------------------------------------------------ quantised trunk linears
 * The trunk is bf16 wherever a recipe left it so; a linear a recipe made int8 (`i8*bf16[1x128]`,
 * rotated `/fwht128` or not, IQ8R) or one of the forms an i8a8h GEMM decodes (the IQR trellis at
 * u5 and its kin, the rotated four- and five-bit asymmetric ones) reads int8 activation codes,
 * which it writes itself from its input's bf16 plane (LinearFP8's op_q: quant_act_i8g, or
 * had_quant_act_i8g for a rotated weight) right in front of its GEMM. No producer of this graph
 * writes codes, so one scratch pair a width K serves every such linear of the rank: [max_tok, K]
 * int8 and its f32 scales, declared at the first one of that K (`Codes`, one a builder; a pair is
 * read at its declared width, so linears of different K cannot share one). Each linear decides
 * by its own encoding (`Lin`), so a bf16 container declares exactly what it did before and any
 * mix of bf16 and int8 linears is served. Other quantised forms (block fp8 or MXFP4, which read
 * E4M3 codes a producer would have to write) are refused by name. */
struct Codes {
    RadBuilder* b = nullptr;
    Names* nm = nullptr;
    int64_t rows = 0;
    struct Pair { int64_t k = 0, g8 = 0; std::string tag; rad_buf q8 = 0, s8 = 0; };
    std::vector<Pair> pairs;
    int n_linear = 0;              /* int8 linears reading a pair */
    int n_fed = 0;                 /* of them, those reading codes another op wrote (Lin::feed) */
    /* The pair at width `k` and scale group `g8` (128, IQ8R 256), declared on first use. A `tag`
     * names a pair of its own, which only the linears and producers of that role write: the codes
     * of a shared input must survive until its last reader (Lin::feed), whatever other linear of
     * the same width runs between. */
    const Pair* get(int64_t k, int64_t g8, const char* tag = "") {
        for (const Pair& p : pairs)
            if (p.k == k && p.g8 == g8 && p.tag == tag) return &p;
        Pair p;
        p.k = k;
        p.g8 = g8;
        p.tag = tag;
        const char* sep = *tag ? "." : "";
        p.q8 = decl_b(b, nm->f("trunk_codes%s%s.%lld.%lld.q8", sep, tag, (long long)k, (long long)g8),
                      RAD_I8, { rows, k });
        p.s8 = decl_b(b, nm->f("trunk_codes%s%s.%lld.%lld.s8", sep, tag, (long long)k, (long long)g8),
                      RAD_F32, { rows, (k + g8 - 1) / g8 });
        if (!p.q8 || !p.s8) return nullptr;
        pairs.push_back(p);
        return &pairs.back();
    }
};
inline std::mutex& codes_mu() {
    static std::mutex m;
    return m;
}
inline std::vector<Codes*>& codes_reg() {
    static std::vector<Codes*> v;
    return v;
}
/* Registers a rank's pair for the duration of its declare (several ranks declare at once). */
struct CodesScope {
    Codes* c;
    explicit CodesScope(Codes* codes) : c(codes) {
        std::lock_guard<std::mutex> g(codes_mu());
        codes_reg().push_back(c);
    }
    ~CodesScope() {
        std::lock_guard<std::mutex> g(codes_mu());
        auto& v = codes_reg();
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i] == c) { v.erase(v.begin() + (std::ptrdiff_t)i); break; }
    }
};
inline Codes* codes_of(RadBuilder* b) {
    std::lock_guard<std::mutex> g(codes_mu());
    for (Codes* c : codes_reg())
        if (c->b == b) return c;
    return nullptr;
}

/* Shared int8 inputs (the friend's q8_fed sharing): several int8 linears read one bf16 input,
 * and the codes the first one's quantiser writes are the codes each of the others would write
 * itself (the same quantiser over the same bytes), so the others read them and declare no
 * quantiser. Byte-identical. A group shares only where the weights read the same codes: the same
 * K, scale group and rotation (Lin::code_form); a member of another form quantises its own. The
 * leader writes a pair of its own (Codes::get's tag), so no linear of the same width between it
 * and the last member overwrites the codes, and the members are issued after it with nothing
 * writing the input between (the caller's order). Off: RADIANCE_HY4_SHARE_CODES=0. */
inline bool share_codes_on() {
    const char* e = std::getenv("RADIANCE_HY4_SHARE_CODES");
    return !(e && *e && std::strcmp(e, "0") == 0);
}
/* The int8 producers that write a linear's codes in their own pass (rmsnorm_had_quant_i8g for the
 * normed q residual, gated_had_quant_i8g for the dense and shared MLPs' down projection), where
 * libr4d serves them. Byte-identical to the producer followed by had_quant_act_i8g (docs/OPS.md).
 * Off: RADIANCE_HY4_FUSED_QUANT=0. */
inline bool fused_quant_on() {
    const char* e = std::getenv("RADIANCE_HY4_FUSED_QUANT");
    return !(e && *e && std::strcmp(e, "0") == 0);
}

/* The chained 128-group rotated int8 quantisers (Geom::chain_q8). Off: RADIANCE_HY4_CHAIN_Q8=0. */
inline bool chain_q8_on() {
    const char* e = std::getenv("RADIANCE_HY4_CHAIN_Q8");
    return !(e && *e && std::strcmp(e, "0") == 0);
}

/* K-19: the routed MoE gather writes the sum's rotated int8 codes and scales in the launch that sums
 * it (moe_gather_inv_i8g), so ffn_latent_out declares no quantiser of its own -- one launch fewer a
 * layer, the same codes. It needs the slot table, no shared arm in the gather and a rotated int8
 * latent_out at a group the latent divides; where those do not hold the plain gather serves and the
 * linear quantises for itself. Off: RADIANCE_HY4_FUSE_GATHER_Q8=0. */
inline bool gather_q8_on() {
    const char* e = std::getenv("RADIANCE_HY4_FUSE_GATHER_Q8");
    return !(e && *e && std::strcmp(e, "0") == 0);
}

/* The G-3 launch fusions (levers-general.md L3), each behind its own switch, on by default:
 * RADIANCE_HY4_FUSE_SWIGLU (the routed arm's clamped SwiGLU with its rotated int8 quantiser),
 * RADIANCE_HY4_FUSE_EXPAND_QUANT (the expansion with the gate and the quantiser of attn_output's
 * input), RADIANCE_HY4_FUSE_JUNCTION_Q (the junction writes the codes its consumers read),
 * RADIANCE_HY4_FUSE_QPROJ (q_nope and q_rope in one launch: dual_fuse_on below). Off with 0; each
 * is byte-identical to the chain it replaces. */
inline bool fuse_on(const char* name) {
    const char* e = std::getenv(name);
    return !(e && *e && std::strcmp(e, "0") == 0);
}

/* K-02's pair launch: a layer's `q_nope` and `q_rope`, both over the one set of codes q_a's
 * quantiser writes, in one `gemm_nt_q_dual` (Lin::declare_dual). Byte-identical to the two
 * launches, and one launch and one ramp cheaper a layer at a decode step, where each of the two
 * moves 0.7-1.3 MB. Off: RADIANCE_HY4_FUSE_QPROJ=0, which keeps the two launches -- P-14: this is
 * the switch fuse_on's list names, now read here. */
inline bool dual_fuse_on() { return fuse_on("RADIANCE_HY4_FUSE_QPROJ"); }

/* The rows a pair's block covers of one part, off that part's own width: the GEMV takes four waves
 * a tile above 4096 columns and eight below (libr4d/r4d_gemm_iqr.hip), and a block of eight waves
 * is then two of its 16-row tiles or one. A part's N must be a whole number of them. */
inline int64_t dual_block_rows(int64_t n) { return n >= 4096 ? 32 : 16; }

/* MoeFP8's fused selection and sort (Config::router_fuse). Off: RADIANCE_HY4_ROUTER_FUSE=0. */
inline bool router_fuse_on() {
    const char* e = std::getenv("RADIANCE_HY4_ROUTER_FUSE");
    return !(e && *e && std::strcmp(e, "0") == 0);
}

/* The int8 codes a stored linear reads: none for a plain bf16 (or unknown) weight, else its K-wise
 * scale group and whether they are in the rotated (Hadamard) domain. */
struct CodeForm {
    bool i8 = false, rot = false;
    int64_t g8 = 0;
    bool same(const CodeForm& o) const { return i8 && o.i8 && rot == o.rot && g8 == o.g8; }
};
inline CodeForm code_form(RadBuilder* b, const char* wn) {
    CodeForm f;
    RadEncoding e{};
    if (!weight_enc(b, wn, &e)) return f;
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return f;
    const bool rot = linear_is_w8h(e) || linear_is_wnh(e);
    if (!rot && !linear_is_w8(e)) return f;
    f.i8 = true;
    f.rot = rot;
    f.g8 = linear_w8h_group(e);
    return f;
}

/* A trunk linear: LinearFP8 at the encoding its weight is stored in (a plain bf16 GEMM, or the
 * int8 forms above over the rank's scratch pair), with LinearFP8's declare and step. */
struct Lin : LinearFP8 {
    rad_buf q8 = 0, s8 = 0;
    int64_t g8 = RAD_FP8_BLOCK;
    CodeForm form{};
    /* Set before declare. `lead`: this linear opens a shared input (a tag naming the role, e.g.
     * "attn_in"), quantising into that role's own pair. `feed`: it reads the codes `feed` wrote
     * (declared and issued before it, over the same input), when their forms match. `fed_codes`:
     * it reads the codes a producer writes in its own pass (`q8`, `s8`, `g8`, `q8_rot`), which the
     * caller declared after checking the form (Lin::code_form). */
    const char* lead = nullptr;
    const Lin* feed = nullptr;
    const ActFP8* fed_codes = nullptr;
    bool fed = false;   /* declared without a quantiser: its codes come from `feed` or `fed_codes` */
    /* The pair's launch (`gemm_nt_q_dual`, K-02): `op_dual` over this Lin's weight and `w2`'s, both
     * products written where their own launches write them -- this Lin's into its `y`, the mate's
     * into `dual_y`. `dual_mate` is the second Lin of a pair: its own `op` is issued above the band
     * the pair serves, and below it the first Lin's launch writes its product. */
    rad_op op_dual = 0;
    rad_weight w2 = 0, ws2 = 0;
    rad_buf dual_y = 0;
    int64_t dual_n = 0;   /* the mate's N: the pair's N2, its own row length */
    bool dual_mate = false;
    /* The heights a pair's launch serves: the GEMV's, one token a step. The pair is two launches'
     * launch and ramp cost, which only the GEMV pays; a wider pass keeps both Lins' launches. */
    static constexpr int64_t kDualMaxM = 1;
    /* This Lin's weight encoding as LinearFP8 read it: a pair's launch reads one form's codes for
     * both weights, so the two must be one encoding -- its planes' runs and row-blocks included,
     * which is what rad_enc_equal sees and what the fold needs to agree on. */
    RadEncoding wenc{};
    ActFP8 with_codes(const ActFP8& a) const {
        ActFP8 r = a;
        if (q8 && !r.q8) {
            r.q8 = q8;
            r.s8 = s8;
            r.g8 = g8;
        }
        if (fed) {
            r.q8 = q8;
            r.s8 = s8;
            r.g8 = g8;
            r.q8_fed = true;
            r.q8_rot = form.rot;
        }
        return r;
    }
    int declare(RadBuilder* b, Names& nm, const Geom& g, const char* base, int64_t n_out,
                int64_t n_in, int shard, RadWeightGroup grp,
                std::initializer_list<const char*> src, const ActFP8& a, rad_buf y,
                int64_t max_rows = 0, std::initializer_list<int64_t> row_parts = {},
                int64_t span_lo = 0, int64_t span_hi = 0) {
        const char* wn = nm.f("%s.weight", base);
        if (!mapped) RAD_ARCH_TRY(map_linear(b, wn, src));   /* a row map is the caller's */
        mapped = true;
        RadEncoding e{};
        const bool known = weight_enc(b, wn, &e);
        const bool plain = !known || (rad_enc_is(&e, "plain") && e.n_planes == 1 &&
                                      e.plane[0].dtype == RAD_BF16);
        Geom gl = g;
        gl.w_bf16 = plain;
        fed = false;
        if (!plain) {
            if (!linear_is_w8(e) && !linear_is_w8h(e) && !linear_is_wnh(e)) {
                char en[256];
                rad_enc_format(&e, en, sizeof en);
                fprintf(stderr, "radiance: hyv4flash: '%s' is %s; a trunk linear is served bf16, "
                                "int8 (i8*bf16[1x128], optionally /fwht128, or IQ8R) or in a form "
                                "the rotated int8 GEMM decodes (the IQR trellis, rotated u4/u5 "
                                "with a zero point)\n", wn, en);
                return RAD_E_DTYPE;
            }
            form = code_form(b, wn);
            wenc = e;
            Codes* c = codes_of(b);
            if (!c) return RAD_E_INVAL;
            if (fed_codes && fed_codes->q8 && fed_codes->s8 && fed_codes->g8 == form.g8 &&
                fed_codes->q8_rot == form.rot) {
                q8 = fed_codes->q8;
                s8 = fed_codes->s8;
                g8 = fed_codes->g8;
                fed = true;
            } else if (feed && feed->lead && feed->q8 && !feed->fed && feed->form.same(form) &&
                       feed->K == n_in &&
                       share_codes_on()) {
                q8 = feed->q8;
                s8 = feed->s8;
                g8 = feed->g8;
                fed = true;
            } else {
                const Codes::Pair* pr =
                    c->get(n_in, form.g8, lead && share_codes_on() ? lead : "");
                if (!pr) return RAD_E_INVAL;
                q8 = pr->q8;
                s8 = pr->s8;
                g8 = pr->g8;
            }
            ++c->n_linear;
            if (fed) ++c->n_fed;
        }
        return LinearFP8::declare(b, nm, gl, base, n_out, n_in, shard, grp, src, with_codes(a), y,
                                  max_rows, row_parts, span_lo, span_hi);
    }
    /* One launch over this Lin's weight and `mate`'s, both over the codes this Lin's input was
     * quantised to (`mate` reads them: Lin::feed). Declared beside both Lins' own ops, taken at the
     * height it serves (kDualMaxM, the GEMV's); above it the two launches run. The pair must be one
     * form -- equal code and scale views and one encoding, run and row-block included, so one decode
     * serves both -- and one K. Its two Ns are its own: each part's blocks are the blocks its own
     * launch gives it, which is what keeps the pair byte-identical from one launch (the row's claim
     * and its oracle's). */
    int declare_dual(RadBuilder* b, Lin& mate, rad_buf x, rad_buf y, rad_buf mate_y, int64_t rows) {
        if (bf16 || mate.bf16 || !q8 || !s8) return RAD_OK;
        if (!iqr || !mate.iqr || !form.rot) return RAD_OK;
        if (!mate.fed || mate.q8 != q8 || mate.s8 != s8 || mate.g8 != g8) return RAD_OK;
        if (mate.K != K || !form.same(mate.form)) return RAD_OK;
        if (!rad_enc_equal(&wenc, &mate.wenc)) return RAD_OK;
        if (!rad_op_declared(b, "gemm_nt_q_dual")) return RAD_OK;
        /* Each part's blocks cover its own tiles: two 16-row tiles where its own launch takes four
         * waves a tile (4096 columns and up), one where it takes eight. The row refuses a part whose
         * N is not a whole number of them, so the pair is not declared here either. */
        if (N % dual_block_rows(N) || mate.N % dual_block_rows(mate.N)) return RAD_OK;
        const RadParam ps[] = { RAD_RANGE("M", 1, rows < kDualMaxM ? rows : kDualMaxM),
                                RAD_INT("N", N), RAD_INT("N2", mate.N), RAD_INT("K", K),
                                RAD_INT("group", g8), RAD_STR("dtype", "i8a8h") };
        const rad_weight wts[] = { w, ws, mate.w, mate.ws };
        op_dual = rw(b, rad_decl_op(b, "gemm_nt_q_dual", ps, 6, wts, 4), { x, q8, s8 },
                     { y, mate_y });
        if (!op_dual) return RAD_E_INVAL;
        w2 = mate.w;
        ws2 = mate.ws;
        dual_y = mate_y;
        dual_n = mate.N;
        mate.dual_mate = true;
        return RAD_OK;
    }
    /* The pair's launch: this Lin's quantiser (it owns the codes then) and one GEMM over both
     * weights into both products. */
    void dual_step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0, int64_t rows,
                   int64_t y0, bool codes_ready) const {
        if (rows <= 0) rows = T - r0;
        const int64_t yr = y0 < 0 ? r0 : y0;
        if (op_q && !codes_ready)
            RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, K), brow_slice(a.q8, r0, rows, K),
                        brow_slice(a.s8, r0, rows, K / a.g8));
        RAD_ISSUE_N(c, op_dual, rows, brow_slice(a.q8, r0, rows, K),
                    brow_slice(a.s8, r0, rows, K / a.g8), RAD_W(w), RAD_W(ws),
                    brow_slice(y, yr, rows, N), RAD_W(w2), RAD_W(ws2),
                    brow_slice(dual_y, yr, rows, dual_n));
    }
    void step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0 = 0, int64_t rows = 0,
              int64_t y0 = -1, bool codes_ready = false) const {
        const int64_t n = rows > 0 || !T ? rows : T - r0;
        /* The pair's second Lin: within the band the first Lin's launch writes this product. */
        if (dual_mate && n <= kDualMaxM) return;
        const ActFP8 in = with_codes(a);
        if (op_dual && n <= kDualMaxM) return dual_step(c, in, y, T, r0, rows, y0, codes_ready);
        LinearFP8::step(c, in, y, T, r0, rows, y0, codes_ready);
    }
};

/* ------------------------------------------------------------------ debug taps
 * RADIANCE_HY4_TAPS=1 (the harness, ref/README.md "Tap contract"): a PERSIST buffer
 * `dbg.<key>.<l>` (`dbg.<key>` for a global key) a tap, [max_tok, width] at the producer's dtype,
 * and a copy into it behind the producing op. Read back through the engine's buffer probe
 * (RADIANCE_DUMP_BUF / RADIANCE_DUMP_DIR). Off, nothing is declared. */
inline bool taps_on() {
    const char* e = std::getenv("RADIANCE_HY4_TAPS");
    return e && *e && std::strcmp(e, "0") != 0;
}

struct Tap {
    rad_buf buf = 0;
    rad_op op = 0;
    int64_t width = 0;
    explicit operator bool() const { return op != 0; }
    /* `src` read [rows, width]; `width` 0 takes the tap's. */
    int declare(RadBuilder* b, Names& nm, int64_t max_rows, const char* key, int layer,
                uint32_t dt, int64_t w, rad_buf src) {
        if (!taps_on()) return RAD_OK;
        width = w;
        const char* name = layer < 0 ? nm.f("dbg.%s", key) : nm.f("dbg.%s.%d", key, layer);
        buf = decl_b(b, name, dt, { max_rows, w }, RAD_BUF_PERSIST);
        if (!buf) return RAD_E_INVAL;
        op = decl_copy(b, max_rows, w, dt, src, buf);
        HY4_NEED(op, "the debug taps' copy (cast)");
        return RAD_OK;
    }
    void step(RadCtx* c, RadOperand src, int64_t rows) const {
        if (op && rows > 0) RAD_ISSUE_N(c, op, rows, src, brows(buf, rows));
    }
};

/* ------------------------------------------------------------------ hyper-connections
 * The independent hyper-connection (HYV4FlashHyperConnection): `fn` scores the flattened streams
 * into 2 * hc logits (the read gate's hc, then the write gate's), `base` their biases, `scale` one
 * factor for each half; ihc_pre collapses the streams into the block's input and gives the write
 * weights, ihc_post adds the block's output back into every stream. f32 in the checkpoint and in
 * the reference. `head` is the final collapse (hc rows, one scale, no write gate). */
struct Hyper {
    rad_weight fn = 0, base = 0, scale = 0;
    rad_op op_pre = 0, op_post = 0;
    bool head = false;
    /* Row strides of the three buffers ihc_post reads, for the row-range form. */
    int64_t hc = 0, n_embd = 0;
    int declare(RadBuilder* b, Names& nm, const Shape& s, int64_t max_tok, const char* scope,
                const char* fn_src, const char* base_src, const char* scale_src,
                RadWeightGroup grp, bool is_head, rad_buf h, rad_buf x, rad_buf post) {
        head = is_head;
        const int64_t rows = head ? s.hc : 2 * s.hc;
        fn = decl_f32(b, nm.f("%s.fn", scope), fn_src, { rows, s.hc * s.n_embd }, grp);
        base = decl_f32(b, nm.f("%s.base", scope), base_src, { rows }, grp);
        scale = decl_f32(b, nm.f("%s.scale", scope), scale_src, { head ? 1 : 2 }, grp);
        if (!fn || !base || !scale) return RAD_E_INVAL;
        if (head) {
            op_pre = mk(b, "ihc_pre",
                        { RAD_RANGE("M", 1, max_tok), RAD_INT("n", s.n_embd), RAD_INT("hc", s.hc),
                          RAD_INT("rows", rows), RAD_F64("eps", s.eps), RAD_F64("hc_eps", s.hc_eps),
                          RAD_STR("dtype", "bf16") },
                        { fn, scale, base }, { h }, { x });
        } else {
            op_pre = mk(b, "ihc_pre",
                        { RAD_RANGE("M", 1, max_tok), RAD_INT("n", s.n_embd), RAD_INT("hc", s.hc),
                          RAD_INT("rows", rows), RAD_F64("eps", s.eps), RAD_F64("hc_eps", s.hc_eps),
                          RAD_F64("post_mag", s.hc_mag), RAD_STR("dtype", "bf16") },
                        { fn, scale, base }, { h }, { x, post });
        }
        HY4_NEED(op_pre, "ihc_pre");
        return RAD_OK;
    }
    /* The write, declared where it is issued (declaration order is liveness): behind the block. */
    int declare_post(RadBuilder* b, const Shape& s, int64_t max_tok, rad_buf h, rad_buf post,
                     rad_buf y) {
        hc = s.hc;
        n_embd = s.n_embd;
        op_post = mk(b, "ihc_post",
                     { RAD_RANGE("M", 1, max_tok), RAD_INT("n", s.n_embd), RAD_INT("hc", s.hc),
                       RAD_STR("dtype", "bf16") },
                     {}, { y, post, h }, { h });
        HY4_NEED(op_post, "ihc_post");
        return RAD_OK;
    }
    void pre(RadCtx* c, rad_buf h, rad_buf x, rad_buf post, int64_t T) const {
        if (head)
            RAD_ISSUE_N(c, op_pre, T, brows(h, T), RAD_W(fn), RAD_W(scale), RAD_W(base),
                        brows(x, T), RAD_NONE);
        else
            RAD_ISSUE_N(c, op_pre, T, brows(h, T), RAD_W(fn), RAD_W(scale), RAD_W(base),
                        brows(x, T), brows(post, T));
    }
    void post_add(RadCtx* c, rad_buf y, rad_buf post, rad_buf h, int64_t T) const {
        RAD_ISSUE_N(c, op_post, T, brows(y, T), brows(post, T), brows(h, T));
    }
    /* Rows [r0, r0 + n): the pipelined prefill writes the connection a piece at a time. */
    void post_add(RadCtx* c, rad_buf y, rad_buf post, rad_buf h, int64_t r0, int64_t n) const {
        RAD_ISSUE_N(c, op_post, n, brow_slice(y, r0, n, n_embd), brow_slice(post, r0, n, hc),
                    brow_slice(h, r0, n, hc * n_embd));
    }
};

/* The buffers every layer shares: the streams, the blocks' inputs and outputs. */
struct Trunk {
    rad_buf h = 0;        /* [T, hc*n] the streams */
    rad_buf x = 0;        /* [T, n] a connection's collapse */
    rad_buf post = 0;     /* [T, hc] f32 its write weights */
    rad_buf xn = 0;       /* [T, n] the block's normed input */
    rad_buf y = 0;        /* [T, n] the block's normed output */
};

/* ------------------------------------------------------------------ the iHC junction (WP16.2)
 * Between two blocks the graph runs `[add] -> [all_reduce] -> rmsnorm(block's post norm) ->
 * ihc_post -> ihc_pre(next connection) -> rmsnorm(next block's input norm)`, five to seven short
 * dependent launches. rmsnorm_ihc_junction is the chain in one (byte-identical to it, docs/OPS.md),
 * ar_rmsnorm_ihc_junction the same behind the exact all-reduce (two ranks at every height, four at
 * decode and verify heights). `tail` false stops behind ihc_post (an Engram sits in front of the
 * next layer and writes h between); `next` the head's collapse makes the last one end in the final
 * norm. y and x are stored only for the debug taps.
 *
 * RADIANCE_HY4_JUNCTION=0 keeps the unfused chain; so does a calibration run (its hooks read
 * between the ops). Where the fused all-reduce is not served (a library without it, four ranks
 * above decode heights, and four ranks at all unless RADIANCE_HY4_JUNCTION_TP4=1) the step issues [add,] the standalone exact all_reduce and the plain row. */
inline bool junction_on() {
    const char* e = std::getenv("RADIANCE_HY4_JUNCTION");
#if defined(_WIN32)
    /* Brilliance: off by default on Windows, the chain the owner's Linux u3 runs used
     * (RADIANCE_HY4_JUNCTION=0); `1` turns the plain row on, and the fused all-reduce twin is not
     * built there (libr4d/r4d_rows.cpp), so the reduction stays Brilliance's checked all_reduce. */
    return e && *e && std::strcmp(e, "0") != 0;
#else
    return !(e && *e && std::strcmp(e, "0") == 0);
#endif
}

/* The latent cache row's front in one launch (mla_row_norm_rope_store, WP16.5): the row's norm, the
 * rotary of its key part and the paged store, byte-identical to the three ops. RADIANCE_HY4_CACHE_ROW=0
 * keeps the three; so do the debug taps (they read between the ops) and a streamed sparse group
 * (its store is kv_store_streamed). */
inline bool cache_row_on() {
    const char* e = std::getenv("RADIANCE_HY4_CACHE_ROW");
    return !(e && *e && std::strcmp(e, "0") == 0);
}

/* The four-rank twin (ar_rmsnorm_ihc_junction _4rank) fuses the reduction into the norm's own
 * kernel, so it holds the CUs while it waits on the link: that pays only where the step is
 * launch-bound. Measured on 4x R9700 behind Gen4 switches at 210 W (Hy4-flash, DFly2 at depth 5,
 * greedy): a six-row verify, one sequence, decodes 132.8 tok/s fused against 131.0 chained, +1.4%;
 * twelve rows (two sequences) 213.1 against 221.6, -3.8%; twenty-four rows -6.2%; forty-eight -9%.
 * kjRows4 is the height up to which it pays: 8 keeps a depth-5 verify fused and stops below two
 * sequences'. World 2 fuses at every height (its twin was tested there).
 * RADIANCE_HY4_JUNCTION_TP4 overrides the four-rank rule: 1 fuses every verify height (the whole
 * ar_wide_write_rows band), 0 never fuses at four ranks, and the chain (add, standalone exact
 * all_reduce, plain junction) carries every height above the band either way. */
enum { kjRows4 = 8 };
inline int64_t junction_rows(const Geom& g) {
    if (g.world == 2) return g.max_tok;      /* tested, and two ranks is one link */
    const char* e = std::getenv("RADIANCE_HY4_JUNCTION_TP4");
    if (e && *e) return std::strcmp(e, "0") == 0 ? 0 : ar_wide_write_rows(g);
    return kjRows4;
}

struct Junction {
    rad_op op = 0, op_ar = 0;           /* the plain row, the all-reduce twin */
    rad_op op_add = 0, op_reduce = 0;   /* the chain's add and all_reduce, for the fallback */
    rad_op op_reduce_ov = 0;            /* the same reduction at `overlap` 1, for the second lane */
    int64_t ar_rows = 0, n = 0, hc = 0, hc_n = 0;
    int world = 1;
    bool tail = true;
    rad_buf a = 0, b = 0, post = 0, h = 0, xn = 0, y = 0, x = 0;
    rad_weight w_post = 0, fn = 0, scale = 0, base = 0, w_next = 0;
    /* G-3b: the rotated int8 codes of the next block's input norm (had_quant_act_i8g's, group 128)
     * this junction writes beside `xn`, so the readers of `xn` issue no quantiser; unset (q8 0) where
     * the chain runs or the junction does not write them. */
    ActFP8 codes{};
    explicit operator bool() const { return op != 0; }
    /* The codes the readers of xn find, or null. */
    const ActFP8* codes_if() const { return op && codes.q8 ? &codes : nullptr; }
    /* Whether the second lane may take this junction's reduction (the narrow twin exists). */
    bool pipe_ok() const { return op_reduce_ov != 0; }

    /* `a` the block's output (`b` the addend in front, 0 for none), `w_post` its sandwich norm's
     * gain; `next` the next connection (0 for tail 0) and `w_next` the next block's input norm,
     * whose output is `xn`. `op_add` / `op_reduce` are the chain's own ops (0 where absent). Leaves
     * `op` 0 (the chain runs) where no library serves the row. `pair` (tail only, G-3b): the code
     * pair the junction writes of `xn` (n % 128 == 0), read by the linears that accept it. */
    int declare(RadBuilder* bl, const Shape& s, const Geom& g, int64_t T, const Trunk& tr,
                rad_buf a_, rad_buf b_, rad_weight w_post_, const Hyper* next, rad_weight w_next_,
                rad_buf xn_, rad_op add_, rad_op reduce_, const Codes::Pair* pair = nullptr) {
        *this = Junction{};
        if (!junction_on() || calib_dir() != nullptr) return RAD_OK;
        if (s.n_embd % 8) return RAD_OK;
        n = s.n_embd;
        hc = s.hc;
        hc_n = s.hc * s.n_embd;
        world = g.world;
        tail = next != nullptr;
        a = a_; b = b_; post = tr.post; h = tr.h;
        w_post = w_post_;
        if (tail) { fn = next->fn; scale = next->scale; base = next->base; w_next = w_next_; xn = xn_; }
        if (taps_on()) { y = tr.y; x = tail ? tr.x : 0; }
        op_add = add_;
        op_reduce = reduce_;
        const int64_t rows = tail && next->head ? s.hc : 2 * s.hc;
        std::vector<RadParam> ps = { RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_INT("hc", s.hc),
                                     RAD_INT("rows", rows), RAD_F64("eps", s.eps),
                                     RAD_F64("ihc_eps", s.eps), RAD_F64("hc_eps", s.hc_eps),
                                     RAD_INT("tail", tail ? 1 : 0), RAD_STR("dtype", "bf16") };
        if (tail && rows == 2 * s.hc) ps.push_back(RAD_F64("post_mag", s.hc_mag));
        std::vector<rad_weight> ws = { w_post };
        if (tail) ws.insert(ws.end(), { fn, scale, base, w_next });
        std::vector<rad_buf> rd = { a, post, h }, wr = { a, post, h };
        if (b) rd.push_back(b);
        if (xn) wr.push_back(xn);
        const bool qc = tail && pair && pair->q8 && pair->s8 && n % 128 == 0 && pair->g8 == 128;
        if (qc) {
            ps.push_back(RAD_INT("group", 128));
            wr.push_back(pair->q8);
            wr.push_back(pair->s8);
        }
        if (y) wr.push_back(y);
        if (x) wr.push_back(x);
        /* An op no loaded library declares (the twin on a host-only deployment) is not declared at
         * all: an undeclared op name stops the startup. */
        auto decl = [&](const char* name, const std::vector<RadParam>& p) -> rad_op {
            if (!rad_op_declared(bl, name)) return 0;
            const rad_op o = rad_decl_op(bl, name, p.data(), (int)p.size(), ws.data(), (int)ws.size());
            if (!o) return o;
            rad_op_reads(bl, o, rd.data(), (int)rd.size());
            rad_op_writes(bl, o, wr.data(), (int)wr.size());
            return o;
        };
        /* The twin first: at more ranks the plain row is the fallback behind the standalone
         * reduction. Two ranks at every height, four up to the height measured to pay (or the
         * whole verify band under RADIANCE_HY4_JUNCTION_TP4). */
        const int64_t jrows = (world == 2 || world == 4) ? junction_rows(g) : 0;
        if (jrows > 0) {
            ar_rows = jrows;
            std::vector<RadParam> pa = ps;
            pa[0] = RAD_RANGE("M", 1, ar_rows);
            pa.push_back(RAD_INT("world_size", world));
            op_ar = decl("ar_rmsnorm_ihc_junction", pa);
        }
        op = decl("rmsnorm_ihc_junction", ps);
        if (!op) op_ar = 0;   /* one served without the other: keep the chain whole */
        if (world > 1 && !op_ar && !op_reduce) op = 0;
        if (op && qc) {
            codes.q8 = pair->q8;
            codes.s8 = pair->s8;
            codes.g8 = 128;
            codes.q8_fed = codes.q8_rot = true;
        }
        /* The narrow twin of the chain's reduction: the pipelined prefill issues it on the second
         * lane beside the GEMMs (a fused `ar_rmsnorm_ihc_junction` would hold its blocks on the
         * card while it waits on the link).
         *
         * Declared wherever the library serves it, whatever the pipeline is set to and whatever
         * `T` is: the arena sizes a level by declaring this graph again at a smaller step, and the
         * builder refuses a probe whose op list differs from the real one op by op (the same
         * buffer, one more reduction per layer). Measured: declared only above the pipeline's row
         * floor, every level below it was dropped ("the sizing declare failed") -- 256 and 1024
         * tokens of an 8192-token deployment -- so small steps could not lend their arena and the
         * boot held 1.9 GiB it did not need. An op that is declared and never issued costs
         * nothing; a graph that changes with `max_tok` costs the loan. */
        if (world > 1 && op_reduce) {
            int rc2 = RAD_OK;
            op_reduce_ov = decl_all_reduce(bl, g, n, a, &rc2, 1);
        }
        return RAD_OK;
    }
    void step(RadCtx* c, int64_t T) const {
        const RadOperand B = b ? brows(b, T) : RAD_NONE;
        auto issue = [&](rad_op o, RadOperand bb) {
            RAD_ISSUE_N(c, o, T, brows(a, T), bb, RAD_W(w_post), brows(post, T), brows(h, T),
                        tail ? RAD_W(fn) : RAD_NONE, tail ? RAD_W(scale) : RAD_NONE,
                        tail ? RAD_W(base) : RAD_NONE, tail ? RAD_W(w_next) : RAD_NONE,
                        xn ? brows(xn, T) : RAD_NONE, y ? brows(y, T) : RAD_NONE,
                        x ? brows(x, T) : RAD_NONE, codes.q8 ? brows(codes.q8, T) : RAD_NONE,
                        codes.s8 ? brows(codes.s8, T) : RAD_NONE);
        };
        if (op_ar && T <= ar_rows) { issue(op_ar, B); return; }
        if (world > 1) {
            if (b) RAD_ISSUE_N(c, op_add, T, brows(a, T), brows(b, T), brows(a, T));
            issue_all_reduce(c, op_reduce, a, T, n);
            issue(op, RAD_NONE);
            return;
        }
        issue(op, B);
    }
    /* Rows [r0, r0 + n) only: what the pipelined prefill hands the second lane, a piece at a time.
     * `beside` picks the narrow reduction (op_reduce_ov), which is the only form allowed on that
     * lane; the fused `op_ar` never runs here (it takes the whole band, so it is only for a small
     * T). Nothing in the chain steps outside a row, so a piece's rows are the same numbers as the
     * whole step's. */
    void step(RadCtx* c, int64_t r0, int64_t rows, bool beside) const {
        if (rows <= 0) return;
        auto row = [&](rad_buf v, int64_t pitch) { return brow_slice(v, r0, rows, pitch); };
        const RadOperand B = b ? row(b, n) : RAD_NONE;
        auto issue = [&](rad_op o, RadOperand bb) {
            RAD_ISSUE_N(c, o, rows, row(a, n), bb, RAD_W(w_post), row(post, hc), row(h, hc_n),
                        tail ? RAD_W(fn) : RAD_NONE, tail ? RAD_W(scale) : RAD_NONE,
                        tail ? RAD_W(base) : RAD_NONE, tail ? RAD_W(w_next) : RAD_NONE,
                        xn ? row(xn, n) : RAD_NONE, y ? row(y, n) : RAD_NONE,
                        x ? row(x, n) : RAD_NONE);
        };
        if (op_ar && rows <= ar_rows && !beside) { issue(op_ar, B); return; }
        if (world > 1) {
            const rad_op red = beside && op_reduce_ov ? op_reduce_ov : op_reduce;
            if (b) RAD_ISSUE_N(c, op_add, rows, row(a, n), row(b, n), row(a, n));
            if (red) RAD_ISSUE_N(c, red, rows * n, row(a, n), RAD_NONE);
            issue(op, RAD_NONE);
            return;
        }
        issue(op, B);
    }
};

}  // namespace hyv4flash
#endif
