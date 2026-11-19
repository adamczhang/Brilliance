/* Brilliance POSIX layer: <sched.h> -- sched_yield only. CPU affinity is not emulated: the
 * Windows answer is processor groups and GetLogicalProcessorInformationEx, and the one site that
 * asks (libavx's physical-core count) says so explicitly. */
#pragma once
#include "brl_compat.h"

BRL_C_BEGIN
int brl_sched_yield(void);
BRL_C_END
#define sched_yield brl_sched_yield
