/* Brilliance POSIX layer: <sys/time.h>. struct timeval comes from winsock2.h, which defines it
 * too; defining our own would collide with any translation unit that also does networking. */
#pragma once
#include "brl_compat.h"
#include <winsock2.h>

BRL_C_BEGIN
int brl_gettimeofday(struct timeval* tv, void* tz);
BRL_C_END
#define gettimeofday brl_gettimeofday
