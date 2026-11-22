/* avx_ggml_hooks.cpp -- the ggml rows' block lookup and layout hooks, compiled once at the
 * baseline (avx_ggml.h says why).
 *
 * The hooks restate libref's (libref/ref_ggml.cpp) over the same libquant code: the selection
 * must be exactly the type's encoding, every plane in order; the stored form is the type's native
 * blocks, packed with pack_of and unpacked with the kind's unpack. Both libraries therefore store
 * the same bytes under the same tag, and a weight relaid for one row reads under the other. */
#include "avx_ggml.h"

#include "rad_encoding.h"
#include "rad_plugin.h"

#include <cstdio>
#include <cstring>

#ifdef AVX_HAVE_GGML
#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"

#include <vector>

namespace {

const lq::ggml::PlaneKind* kind_named(const char* s) {
    if (!s) return nullptr;
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (k && !std::strcmp(k->name, s) && lq::ggml::pack_of(t) && lq::ggml::dequant_of(t)) return k;
    }
    return nullptr;
}

int64_t pint(const RadParam* p, int n_p, const char* key) {
    const RadParam* q = rad_param_find(p, n_p, key);
    if (!q) return 0;
    if (q->kind == RAD_P_RANGE) return (int64_t)q->ihi;
    return q->kind == RAD_P_INT ? (int64_t)q->ival : 0;
}

const lq::ggml::PlaneKind* kind_of(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return kind_named(t && t->kind == RAD_P_STR ? t->sval : nullptr);
}

int ggml_check(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
               int n_planes, const lq::ggml::PlaneKind** out) {
    const lq::ggml::PlaneKind* k = kind_of(p, n_p);
    if (operand != 1 || !k || !enc || !sel) return RAD_E_DTYPE;
    const RadEncoding want = lq::ggml::encoding_of(k->type);
    if (!rad_enc_equal(enc, &want) || n_planes != want.n_planes) return RAD_E_DTYPE;
    for (int i = 0; i < n_planes; ++i)
        if (sel[i] != i) return RAD_E_SHAPE;
    *out = k;
    return RAD_OK;
}

uint8_t* plane_row(const RadEncoding* enc, const RadTensor* planes, int i, int64_t r) {
    const RadEncPlane& pl = enc->plane[i];
    if (pl.kind == RAD_PLANE_TABLE || (pl.block[0] == 0 && pl.block[1] == 0)) return nullptr;
    const int64_t cols = planes[i].rank >= 2 ? planes[i].shape[1] : 0;
    return (uint8_t*)planes[i].data + r * rad_enc_row_bytes(pl.dtype, cols);
}

int64_t logical_rows(const RadTensor* planes) { return planes[0].rank >= 1 ? planes[0].shape[0] : 0; }

int64_t row_len(const RadParam* p, int n_p) {
    const int64_t K = pint(p, n_p, "K");
    return K > 0 ? K : pint(p, n_p, "n_embd");
}

}  // namespace

extern "C" int avx_ggml_type(const char* name, int64_t* block, int64_t* block_bytes,
                             AvxGgmlDequant* dequant) {
    const lq::ggml::PlaneKind* k = kind_named(name);
    if (!k) return 0;
    if (block) *block = k->block;
    if (block_bytes) *block_bytes = (int64_t)k->block_bytes;
    if (dequant) *dequant = lq::ggml::dequant_of(k->type);
    return 1;
}

extern "C" int avx_ggml_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const RadTensor* planes, int n_planes, RadLayout* out) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = row_len(p, n_p), rows = logical_rows(planes);
    if (K <= 0 || K % k->block || rows <= 0) return RAD_E_SHAPE;
    static thread_local char tag[32];
    std::snprintf(tag, sizeof tag, "ggml_%s", k->name);
    std::memset(out, 0, sizeof *out);
    out->tag = tag;
    out->dtype = RAD_U8;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = K / k->block * (int64_t)k->block_bytes;
    out->align = 64;
    out->bytes = out->shape[0] * out->shape[1];
    return RAD_OK;
}

extern "C" int avx_ggml_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* planes, int n_planes, void* dst,
                                 int64_t dst_bytes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = row_len(p, n_p), rows = logical_rows(planes);
    const int64_t nb = K > 0 ? K / k->block : 0, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || dst_bytes < rows * rb) return RAD_E_SHAPE;
    const lq::ggml::PackFn pack = lq::ggml::pack_of(k->type);
    std::vector<const uint8_t*> row((size_t)n_planes);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = plane_row(enc, planes, i, r);
        pack(row.data(), nb, (uint8_t*)dst + r * rb);
    }
    return RAD_OK;
}

extern "C" int avx_ggml_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const void* src, int64_t src_bytes,
                                   const RadTensor* planes, int n_planes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = row_len(p, n_p), rows = logical_rows(planes);
    const int64_t nb = K > 0 ? K / k->block : 0, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || src_bytes < rows * rb) return RAD_E_SHAPE;
    std::vector<uint8_t*> row((size_t)n_planes);
    std::vector<void*> all((size_t)n_planes);
    for (int i = 0; i < n_planes; ++i) all[(size_t)i] = planes[i].data;
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = plane_row(enc, planes, i, r);
        k->unpack((const uint8_t*)src + r * rb, nb, row.data());
    }
    lq::ggml::write_shared(k->type, *enc, all.data());
    return RAD_OK;
}

#else  /* !AVX_HAVE_GGML: no native reader; the registry declares no ggml row. */

extern "C" int avx_ggml_type(const char*, int64_t*, int64_t*, AvxGgmlDequant*) { return 0; }
extern "C" int avx_ggml_layout(const RadParam*, int, int, const RadEncoding*, const int*,
                               const RadTensor*, int, RadLayout*) { return RAD_E_UNSUPPORTED; }
extern "C" int avx_ggml_relayout(const RadParam*, int, int, const RadEncoding*, const int*,
                                 const RadTensor*, int, void*, int64_t) { return RAD_E_UNSUPPORTED; }
extern "C" int avx_ggml_unrelayout(const RadParam*, int, int, const RadEncoding*, const int*,
                                   const void*, int64_t, const RadTensor*, int) {
    return RAD_E_UNSUPPORTED;
}

#endif
