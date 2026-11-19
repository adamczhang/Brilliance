/* Brilliance POSIX layer: <sys/syscall.h> is Linux's raw system-call table. Nothing in it exists
 * on Windows, so this header is empty and every SYS_* use stays behind the __linux__ guards
 * radiance already has around it. */
#pragma once
#include "brl_compat.h"
