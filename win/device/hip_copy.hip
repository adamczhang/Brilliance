/* hip_copy.hip -- Brilliance: the ranged copy behind rad_memcpy_ranges_async as one kernel on the
 * stream's own compute queue, as on Linux (core/device/aql_blit.hip, aql_copy_ranges, which this
 * copies; that file is left out of the Windows build with the rest of the AQL backend).
 *
 * WHY A KERNEL AND NOT hipMemcpyAsync. The step batch's staging slab on the card is one per rank and
 * never double buffered (sched/batch.h, Set): stream order alone keeps a build's upload from
 * overwriting what the passes before it are still reading. When the runtime's copy engines (SDMA)
 * carry hipMemcpyAsync, that order does not hold on gfx1201 with HIP 7.16: an upload issued behind
 * hipEventRecord (every build records its flight event) can run before the kernels ahead of it on
 * the stream have read the slab (rocm-systems#12571's class of fault, here write-after-read). The
 * engine then advances a verify step from the next pass's row descriptors -- a drafter's acceptance
 * read as 0 where the host counted 7 (specdec-sdma, 2026-11-20) -- and stops on the disagreement.
 * A kernel on the compute queue is ordered with the kernels around it by the queue itself,
 * whichever engine carries the runtime's copies.
 *
 * The source is pinned host memory or device memory, the range list pinned host memory the card
 * reads when the kernel runs (rad_memcpy_ranges_async's contract: unchanged until the stream passes
 * the copy). The first `prefix_blocks` workgroups copy the prefix in 16-byte words; each later
 * workgroup copies one range in 4-byte words, 16 a thread a round with every load issued before
 * the first store, and loads past the end clamped rather than guarded. */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "device/stream_card.h"   /* a launch from its stream's card */
#include "device/device.h"         /* RadCopyRange */

namespace {

constexpr int kThreads = 256;
constexpr uint64_t kMaxBlocks = 2048;
constexpr int kRangeWords = 16;

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(kThreads) void brl_copy_ranges(const u32x4* __restrict__ src16,
                                                            u32x4* __restrict__ dst16, uint64_t n16,
                                                            unsigned prefix_blocks,
                                                            const RadCopyRange* __restrict__ r) {
    if (blockIdx.x < prefix_blocks) {
        const uint64_t stride = (uint64_t)prefix_blocks * kThreads;
        uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
        for (; i + 3 * stride < n16; i += 4 * stride) {
            const u32x4 a = src16[i], b = src16[i + stride], c = src16[i + 2 * stride],
                        d = src16[i + 3 * stride];
            dst16[i] = a; dst16[i + stride] = b; dst16[i + 2 * stride] = c; dst16[i + 3 * stride] = d;
        }
        for (; i < n16; i += stride) dst16[i] = src16[i];
        return;
    }
    const RadCopyRange g = r[blockIdx.x - prefix_blocks];
    const unsigned* s4 = (const unsigned*)((const char*)src16 + g.src);
    unsigned* d4 = (unsigned*)((char*)dst16 + g.dst);
    const int64_t n = g.bytes / 4;
    if (n <= 0) return;
    for (int64_t base = 0; base < n; base += (int64_t)kThreads * kRangeWords) {
        unsigned v[kRangeWords];
#pragma unroll
        for (int k = 0; k < kRangeWords; ++k) {
            const int64_t w = base + (int64_t)k * kThreads + threadIdx.x;
            v[k] = s4[w < n ? w : n - 1];
        }
#pragma unroll
        for (int k = 0; k < kRangeWords; ++k) {
            const int64_t w = base + (int64_t)k * kThreads + threadIdx.x;
            if (w < n) d4[w] = v[k];
        }
    }
}

}  // namespace

/* One dispatch onto `stream`: the prefix (a multiple of 16 bytes, both bases 16-byte aligned), then
 * the `n` ranges. 0, or -1 with the launch error left for the caller (hipGetLastError). */
extern "C" int brl_hip_copy_ranges(void* dst, const void* src, uint64_t prefix,
                                   const RadCopyRange* ranges, int n, void* stream) {
    const uint64_t n16 = prefix / 16;
    uint64_t pb = n16 ? (n16 + 4 * kThreads - 1) / (4 * kThreads) : 0;
    if (pb > kMaxBlocks) pb = kMaxBlocks;
    if (pb == 0 && n <= 0) return 0;
    rad::OnStreamCard on((void*)stream);
    hipLaunchKernelGGL(brl_copy_ranges, dim3((unsigned)pb + (unsigned)(n > 0 ? n : 0)), dim3(kThreads),
                       0, (hipStream_t)stream, (const u32x4*)src, (u32x4*)dst, n16, (unsigned)pb,
                       ranges);
    return hipPeekAtLastError() == hipSuccess ? 0 : -1;
}

namespace {

__global__ __launch_bounds__(kThreads) void brl_copy16(const u32x4* __restrict__ src,
                                                       u32x4* __restrict__ dst, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

__global__ __launch_bounds__(kThreads) void brl_copy1(const unsigned char* __restrict__ src,
                                                      unsigned char* __restrict__ dst, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

__global__ __launch_bounds__(kThreads) void brl_copy2d1(const unsigned char* __restrict__ src,
                                                        unsigned char* __restrict__ dst,
                                                        uint64_t spitch, uint64_t dpitch,
                                                        uint64_t width, uint64_t height) {
    const uint64_t cstride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t r = blockIdx.y; r < height; r += gridDim.y) {
        const unsigned char* s = src + r * spitch;
        unsigned char* d = dst + r * dpitch;
        for (uint64_t c = (uint64_t)blockIdx.x * kThreads + threadIdx.x; c < width; c += cstride)
            d[c] = s[c];
    }
}

inline unsigned blocks_for(uint64_t n, uint64_t per_block) {
    uint64_t b = (n + per_block - 1) / per_block;
    if (b < 1) b = 1;
    if (b > kMaxBlocks) b = kMaxBlocks;
    return (unsigned)b;
}

}  // namespace

/* A small copy as a kernel on `stream` (hipnative's n_memcpy_async while the copy engines are on):
 * both ends memory the card reaches, device or pinned host. 0, or -1 with the launch error left. */
extern "C" int brl_hip_copy(void* dst, const void* src, uint64_t bytes, void* stream) {
    if (bytes == 0) return 0;
    rad::OnStreamCard on(stream);
    const hipStream_t s = (hipStream_t)stream;
    if ((((uintptr_t)dst | (uintptr_t)src | bytes) & 15u) == 0) {
        const uint64_t n = bytes / 16;
        hipLaunchKernelGGL(brl_copy16, dim3(blocks_for(n, kThreads)), dim3(kThreads), 0, s,
                           (const u32x4*)src, (u32x4*)dst, n);
    } else {
        hipLaunchKernelGGL(brl_copy1, dim3(blocks_for(bytes, kThreads)), dim3(kThreads), 0, s,
                           (const unsigned char*)src, (unsigned char*)dst, bytes);
    }
    return hipPeekAtLastError() == hipSuccess ? 0 : -1;
}

/* And a small pitched copy, the same way. */
extern "C" int brl_hip_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                              uint64_t width, uint64_t height, void* stream) {
    if (width == 0 || height == 0) return 0;
    rad::OnStreamCard on(stream);
    const unsigned by = (unsigned)(height < 65535 ? height : 65535);
    hipLaunchKernelGGL(brl_copy2d1, dim3(blocks_for(width, kThreads), by), dim3(kThreads), 0,
                       (hipStream_t)stream, (const unsigned char*)src, (unsigned char*)dst, spitch,
                       dpitch, width, height);
    return hipPeekAtLastError() == hipSuccess ? 0 : -1;
}
