/* wtab.hip -- the weight-table patch kernel: one thread a write. See wtab.h. */
#include <hip/hip_runtime.h>
#if defined(_WIN32)
#include "device/stream_card.h"   /* Brilliance: a launch from its stream's card */
#endif

#include "wtab.h"

namespace {

__global__ __launch_bounds__(64) void rad_wtab_patch(rad::WtabPatch p) {
    const uint32_t i = threadIdx.x;
    if (i < p.n) *p.dst[i] = p.val[i];
}

}  // namespace

extern "C" int rad_wtab_patch_launch(const rad::WtabPatch* p, void* stream) {
    if (!p || p->n == 0) return 0;
    #if defined(_WIN32)
    const rad::OnStreamCard on_card((void*)stream);   /* Brilliance: win/device/stream_card.h */
    #endif
    hipLaunchKernelGGL(rad_wtab_patch, dim3(1), dim3(64), 0, (hipStream_t)stream, *p);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
