# brl_identity.cmake -- Brilliance: writes OUT, the header that names this build (cmake -P, at every
# build; win/cmake/RadWindows.cmake, brl_windows_finalize):
#   BRILLIANCE_VERSION   the release (RadWindows.cmake's BRILLIANCE_VERSION)
#   BRILLIANCE_COMMIT    the source commit, 12 hex digits, "-dirty" when tracked files differ from it
#   BRILLIANCE_UPSTREAM  the upstream tag and commit the port is pinned to
# The file is rewritten only when that text changes, so a build at the same commit recompiles nothing.
#
#   cmake -DSRC=<checkout> -DOUT=<header> -DVERSION=<release> -DUPSTREAM=<tag sha> -P brl_identity.cmake
set(_sha "unknown")
execute_process(COMMAND git -C "${SRC}" rev-parse --short=12 HEAD
                OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rc ERROR_QUIET)
if(_rc EQUAL 0 AND NOT _head STREQUAL "")
  set(_sha "${_head}")
  execute_process(COMMAND git -C "${SRC}" status --porcelain --untracked-files=no
                  OUTPUT_VARIABLE _st RESULT_VARIABLE _rc ERROR_QUIET)
  if(_rc EQUAL 0 AND NOT _st STREQUAL "")
    string(APPEND _sha "-dirty")
  endif()
endif()
string(CONCAT _body
       "/* Generated at build time by win/cmake/brl_identity.cmake, not in the source tree. */\n"
       "#pragma once\n"
       "#define BRILLIANCE_VERSION \"${VERSION}\"\n"
       "#define BRILLIANCE_COMMIT \"${_sha}\"\n"
       "#define BRILLIANCE_UPSTREAM \"${UPSTREAM}\"\n")
set(_old "")
if(EXISTS "${OUT}")
  file(READ "${OUT}" _old)
endif()
if(NOT _old STREQUAL _body)
  file(WRITE "${OUT}" "${_body}")
endif()
