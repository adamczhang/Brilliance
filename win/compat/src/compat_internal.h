// Brilliance POSIX layer: what the emulations share and the public headers do not declare.
#pragma once

#include <windows.h>
#include <stdint.h>
#include <stdlib.h>

namespace brl {

// A BAD DESCRIPTOR IS AN ERROR VALUE HERE, NOT THE END OF THE PROCESS. The UCRT checks every fd it
// is handed (_get_osfhandle, _read, _write, _commit, _lseeki64) and on a bad one calls the
// invalid-parameter handler, whose default ends the process with 0xC0000417 -- where POSIX returns
// -1 with EBADF, which is what the callers of these functions test for. The handler is replaced for
// the duration of one CRT call and on this thread only: installing a process-wide one would take
// the fast-fail away from every other CRT call in the process, including the ones in code that is
// not ours (the HIP runtime, a plugin), for which a bad parameter is a bug worth stopping on. With
// the handler returning, the CRT function returns its documented failure (-1, errno EBADF/EINVAL).
class QuietCrt {
public:
    QuietCrt() : prev_(_set_thread_local_invalid_parameter_handler(ignore)) {}
    ~QuietCrt() { _set_thread_local_invalid_parameter_handler(prev_); }
    QuietCrt(const QuietCrt&) = delete;
    QuietCrt& operator=(const QuietCrt&) = delete;

private:
    static void __cdecl ignore(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {}
    _invalid_parameter_handler prev_;
};

// The Win32 handle behind a CRT descriptor, or INVALID_HANDLE_VALUE with errno EBADF.
HANDLE osfhandle(int fd);

}   // namespace brl
