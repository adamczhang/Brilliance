/* Brilliance POSIX layer: <sys/sysmacros.h>. Device numbers have no Windows meaning; the only
 * user is radiance's /proc scan for other processes holding a file, which is Linux-only. */
#pragma once
#include "brl_compat.h"
#define major(d) ((unsigned)(0))
#define minor(d) ((unsigned)(0))
