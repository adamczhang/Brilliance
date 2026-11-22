/* avx_ggml.h -- llama.cpp's native blocks for the ggml rows (gemm_nt_ggml, moe_gemm_ggml,
 * logits_gemm_ggml, embed_lookup_ggml), split as avx_rows.h splits the n-gram gather.
 *
 * The blocks' one definition is libquant's (lq_ggml_planes, lq_ggml_dequant), the code libref
 * and the core compile. libavx compiles it once, at the baseline (avx_ggml_hooks.cpp), with the
 * rows' layout hooks; the per-level kernels (avx_ggml.cpp) reach it only through the plain
 * declarations below, so no inline function of libquant's headers is compiled at an ISA level and
 * merged by the linker into another level's table (avx_vec.h says why that matters).
 *
 * Built only when libquant's sources are beside libavx's (AVX_HAVE_GGML), as libref's rows are:
 * out of tree the ggml rows are absent from both libraries. */
#ifndef RAD_AVX_GGML_H
#define RAD_AVX_GGML_H

#include "rad_abi.h"

#include <cstdint>

extern "C" {

/* `k` values (whole blocks) of one row's native blocks -> f32, llama.cpp's dequantize_row. */
typedef void (*AvxGgmlDequant)(const void* blocks, float* y, int64_t k);

/* The GGUF type llama.cpp spells `name` ("iq1_s", "q6_K", ...): the values a block covers, its
 * bytes and its dequantiser. Returns 0 for a type no native reader exists for, or always without
 * AVX_HAVE_GGML. Compiled at the baseline. */
int avx_ggml_type(const char* name, int64_t* block, int64_t* block_bytes, AvxGgmlDequant* dequant);

/* The rows' layout hooks, libref's (libref/ref_ggml.cpp) restated over the same libquant code:
 * the type's planes in, the type's native blocks [rows, K / block * block_bytes] u8 stored, so
 * both libraries store the same bytes under one tag. */
int avx_ggml_layout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                    int, RadLayout*);
int avx_ggml_relayout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                      int, void*, int64_t);
int avx_ggml_unrelayout(const RadParam*, int, int, const RadEncoding*, const int*, const void*,
                        int64_t, const RadTensor*, int);

}  /* extern "C" */

#endif /* RAD_AVX_GGML_H */
