/* Brilliance: libr4d's own log lines, host code only.
 *
 * r4d_log (r4d_rows.cpp) writes one line through the host's logger when the loader handed one over
 * (rad_plugin_set_log, abi/rad_abi.h), so the line is level-filtered and tagged like the core's and
 * a W line counts into the startup report's FINDINGS. A library never handed one (r4d_selftest
 * dlopens it itself) prints "<L> <text>" to stderr as before. `level` is 0 E, 1 W, 2 I, 3 D, 4 T;
 * the text carries no level tag and no final newline ("r4d: ..."), and may be of any length.
 *
 * r4d_ar_env_read (r4d_ar_entry.hip) reads the collectives' environment switches --
 * BRILLIANCE_MAILBOX_MEM, BRILLIANCE_AR_RESEND, BRILLIANCE_AR_INJECT -- once, as rad_plugin_open
 * reads BRILLIANCE_EXACT_VERIFY, so a bad or unusual value is logged at start, before the startup
 * report counts the W lines, rather than at the first collective that asks. */
#pragma once

void r4d_log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

extern "C" void r4d_ar_env_read(void);
