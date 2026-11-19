/* Brilliance POSIX layer: <sys/wait.h> -- the status macros only.
 *
 * There is no fork on Windows and no waitpid; a site that starts a process uses
 * os::spawn_and_wait (core/os/os.h), which reports an exit code or, for a crash, the NTSTATUS
 * (0xC0000005 and friends). These macros read such a status the way they read a POSIX one, so
 * code that only interprets a status compiles unchanged: a status with the high bit set -- an
 * NTSTATUS of error (0xC...) or warning (0x8...) severity, which is how an unhandled exception, a
 * breakpoint (0x80000003) or a fast-fail ends a process -- is "signalled", and WTERMSIG names the
 * POSIX signal that ends a Linux process the same way, so `128 + WTERMSIG(s)` and a switch over
 * SIGSEGV/SIGABRT/... read as they do there. A status no signal describes reads as SIGABRT,
 * abnormal termination. WEXITSTATUS keeps the whole status, for a message that wants the code. */
#pragma once
#include "brl_compat.h"
#include <signal.h>

#define WNOHANG 1

/* The POSIX signals Windows' <signal.h> lacks, only ever compared against (spawn.h has the same). */
#ifndef SIGTRAP
#define SIGTRAP 5
#endif
#ifndef SIGBUS
#define SIGBUS 7
#endif
#ifndef SIGKILL
#define SIGKILL 9
#endif

BRL_C_BEGIN
/* A child started by posix_spawn (spawn.h); `options` may be WNOHANG. */
pid_t waitpid(pid_t pid, int* status, int options);

static inline int brl_wtermsig(int status) {
    switch ((unsigned)status) {
    case 0xC0000005u:                       /* access violation */
    case 0xC00000FDu:                       /* stack overflow */
        return SIGSEGV;
    case 0xC0000006u:                       /* in-page error: a mapped file could not be read */
    case 0x80000002u:                       /* datatype misalignment */
        return SIGBUS;
    case 0xC000001Du:                       /* illegal instruction */
    case 0xC0000096u:                       /* privileged instruction */
        return SIGILL;
    case 0xC000008Du: case 0xC000008Eu: case 0xC000008Fu: case 0xC0000090u:
    case 0xC0000091u: case 0xC0000092u: case 0xC0000093u:   /* floating-point faults */
    case 0xC0000094u: case 0xC0000095u:                     /* integer divide by zero, overflow */
    case 0xC00002B4u: case 0xC00002B5u:                     /* multiple float faults or traps */
        return SIGFPE;
    case 0x80000003u:                       /* breakpoint: __debugbreak, a trap instruction */
    case 0x80000004u:                       /* single step */
        return SIGTRAP;
    case 0xC000013Au:                       /* Ctrl-C */
        return SIGINT;
    default:                                /* 0xC0000409 fast-fail (abort), and the rest */
        return SIGABRT;
    }
}
BRL_C_END

#define WIFEXITED(s)   ((((unsigned)(s)) & 0x80000000u) == 0)
#define WEXITSTATUS(s) ((int)(s))
#define WIFSIGNALED(s) (!WIFEXITED(s))
#define WTERMSIG(s)    brl_wtermsig((int)(s))
