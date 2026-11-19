/* Brilliance POSIX layer: <pthread.h>, thread naming only.
 *
 * radiance creates its threads with std::thread and uses pthreads for one thing: naming the
 * calling thread for a debugger or profiler (pthread_setname_np(pthread_self(), "rad-...")). That
 * carries over exactly -- SetThreadDescription names a thread for the same tools -- so it is here.
 * Nothing else of pthreads is: a call to any other pthread function is a compile error at the
 * site, which is where the decision about its Windows meaning belongs. */
#pragma once
#include "brl_compat.h"

typedef void* pthread_t;   /* a thread handle; pthread_self's is the calling thread's pseudo-handle */

BRL_C_BEGIN
pthread_t pthread_self(void);
/* Names the thread (UTF-8). Returns 0, or an errno value as pthreads does. */
int       pthread_setname_np(pthread_t thread, const char* name);
BRL_C_END
