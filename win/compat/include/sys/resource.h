/* Brilliance POSIX layer: <sys/resource.h>.
 *
 * Windows has no resource limits in the rlimit sense. The one radiance asks about is
 * RLIMIT_MEMLOCK, and the Windows bound on locked memory is the process's minimum working set,
 * which the engine raises itself before it locks (core/os). So the honest answer is "unlimited":
 * a limit radiance could act on does not exist here. */
#pragma once
#include "brl_compat.h"

typedef unsigned long long rlim_t;
struct rlimit { rlim_t rlim_cur, rlim_max; };

#define RLIM_INFINITY  (~(rlim_t)0)
#define RLIMIT_MEMLOCK 8
#define RLIMIT_NOFILE  7
#define RLIMIT_STACK   3
#define RLIMIT_AS      9

BRL_C_BEGIN
int getrlimit(int resource, struct rlimit* rl);
BRL_C_END
