/* Brilliance POSIX layer: <spawn.h> -- posix_spawn over CreateProcessW, for the harnesses that run
 * each case in a child of its own (libr4d's r4d_selftest).
 *
 * Supported: a path, an argv, an envp, and file actions that dup2 a descriptor onto the child's
 * standard input, output or error (0, 1 or 2) -- what r4d_selftest uses to give each row's child a
 * buffer of its own. No other file action and no attributes (attr must be null). The child's pid
 * is its Windows process id; waitpid() (sys/wait.h) waits on it, or on any child for pid -1, and
 * reports its exit code, or for a crash the NTSTATUS (0xC0000005 ...), which WIFSIGNALED reads as
 * "signalled". */
#pragma once
#include "brl_compat.h"

typedef void* posix_spawn_file_actions_t;
typedef void* posix_spawnattr_t;

/* The signals POSIX code names when it reports how a child died; Windows' <signal.h> lacks these
 * two. They are only ever compared against, never raised. */
#ifndef SIGBUS
#define SIGBUS 7
#endif
#ifndef SIGKILL
#define SIGKILL 9
#endif

BRL_C_BEGIN
int posix_spawn_file_actions_init(posix_spawn_file_actions_t* fa);
int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t* fa);
/* newfd 0, 1 or 2 only (EINVAL otherwise). */
int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t* fa, int fd, int newfd);
int posix_spawn(pid_t* pid, const char* path, const posix_spawn_file_actions_t* actions,
                const posix_spawnattr_t* attr, char* const argv[], char* const envp[]);
int posix_spawnp(pid_t* pid, const char* file, const posix_spawn_file_actions_t* actions,
                 const posix_spawnattr_t* attr, char* const argv[], char* const envp[]);
BRL_C_END
