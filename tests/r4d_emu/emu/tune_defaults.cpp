/* The row table's tuned axes, for the one emu unit that reads them.
 *
 * An emu unit is a libr4d source compiled on the host. r4d_moe.hip publishes GLM's router and
 * scatter rows as well as its gather, and a row's args read the tuned configuration through
 * r4d_cfg_gemm (libr4d/r4d_rows.cpp). The RadTunable table is not part of the emulator, and the
 * cases here launch that file's gather only, so no axis is ever read: this answers with the
 * untuned defaults the table declares (tGemmSkinny: wv 1, sk 0, mb 0, npw 1, nt 1), which is what
 * an install with no tuned key runs. A GEMM case added to the emu would need the real table. */
#include "r4d_args.h"

#include <cstring>

extern "C" void r4d_cfg_gemm(const RadArgs*, R4dGemmCfg* out) {
    std::memset(out, 0, sizeof(*out));
    out->wv = 1;
    out->npw = 1;
    out->nt = 1;
}
