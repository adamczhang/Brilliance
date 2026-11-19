/* Brilliance POSIX layer: <dlfcn.h> over LoadLibraryExW / GetProcAddress.
 *
 * The meaning carries over: a plugin is a module loaded by path and its C entry points are
 * looked up by name. A DLL is always "local" in the RTLD_LOCAL sense (its symbols never satisfy
 * another module's imports), which is what radiance's loader asks for anyway.
 *
 * Paths are UTF-8 and converted to UTF-16 for the loader. A name containing a separator is a path,
 * relative to the current directory as dlopen has it, and is made absolute before it is loaded with
 * LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, so a plugin's own
 * dependencies resolve from its directory, the exe's directory (where the HIP runtime and libomp
 * are staged) and System32, and never from the current directory or PATH order accidents. A bare
 * name is searched for in the exe's directory, System32 and AddDllDirectory's, never the current
 * directory, as dlopen searches its library path and not ".".
 * dlerror() reports the Win32 message of the last failure on this thread. */
#pragma once
#include "brl_compat.h"

#define RTLD_LAZY     0x0001
#define RTLD_NOW      0x0002
#define RTLD_GLOBAL   0x0100
#define RTLD_LOCAL    0x0000
#define RTLD_NOLOAD   0x0004
#define RTLD_NODELETE 0x1000

/* RTLD_DEFAULT: the executable, then every module loaded in the process. RTLD_NEXT has no
 * Windows meaning (there is no interposition) and is not defined, so a use is a compile error. */
#define RTLD_DEFAULT ((void*)0)

BRL_C_BEGIN
void*       dlopen(const char* path, int flags);
void*       dlsym(void* handle, const char* name);
int         dlclose(void* handle);
char*       dlerror(void);

typedef struct {
    const char* dli_fname;
    void*       dli_fbase;
    const char* dli_sname;
    void*       dli_saddr;
} Dl_info;
/* The module containing `addr`; the symbol fields are always null. */
int dladdr(const void* addr, Dl_info* info);
BRL_C_END
