# Brilliance: everything the Windows build does differently, in one file.
#
# Included from the top-level CMakeLists.txt right after project() on WIN32; brl_windows_finalize()
# runs at the end of the top-level file. The plugin half -- the compiler check, the flags every
# translation unit takes, OpenMP and brl_win_plugin, which cmake/RadAddPlugin.cmake calls -- is
# RadWindowsPlugin.cmake, included from here and installed with the package config for plugins
# built against an installed radiance. Kept out of radiance's own CMakeLists so a later radiance
# release merges against a handful of one-line hooks rather than Windows logic spread through every
# file. Adapted from Brilliance 0.2.2's cmake/BrillianceWindows.cmake.

include_guard(GLOBAL)

# Brilliance's own release: what the package name carries. The project() version stays radiance's,
# the release this tree was taken from.
set(BRILLIANCE_VERSION "0.3.0-alpha.3" CACHE INTERNAL "Brilliance's release")
# The upstream revision the port is on, as `git describe --tags` names it, and its commit; an upstream
# merge moves it. Since the merge of upstream's v2.2.0 tag (b6b8751c, 2026-11-19) it is that release.
set(BRILLIANCE_UPSTREAM "v2.2.0 b6b8751cccb0e834a13ae9e5afca824ee9a487aa" CACHE INTERNAL
    "the upstream tag and commit Brilliance is pinned to")

# An install goes to Program Files\Brilliance unless the configure named a prefix; CMake's own
# default is Program Files (x86)\<project>, the 32-bit directory under radiance's name.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  file(TO_CMAKE_PATH "$ENV{ProgramW6432}/Brilliance" _brl_prefix)
  set(CMAKE_INSTALL_PREFIX "${_brl_prefix}" CACHE PATH "install prefix" FORCE)
endif()
include(GNUInstallDirs)

# ------------------------------------------------------------------------------- the compiler
# The compiler check, the definitions and warning flags, the shared CRT, compiler-rt's builtins and
# OpenMP: what a plugin built out of the tree needs as much as the tree does, so they live in the
# plugin half (RadWindowsPlugin.cmake), which the tree applies to every target in its directories.
set(BRL_IN_TREE ON)
include("${CMAKE_CURRENT_LIST_DIR}/RadWindowsPlugin.cmake")
add_compile_definitions(${BRL_WIN_DEFINITIONS})
add_compile_options(${BRL_WIN_CXX_OPTIONS})

# The plugin half and the delay-load hook it compiles into every plugin, installed beside
# RadAddPlugin.cmake; the installed radiance-config.cmake includes the plugin half on Windows.
install(FILES "${CMAKE_CURRENT_LIST_DIR}/RadWindowsPlugin.cmake"
              "${CMAKE_CURRENT_LIST_DIR}/rad_delayload_hook.cpp"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radiance)

# EIGHT MiB STACKS, as Linux gives every thread. Windows defaults to 1 MiB, and the jinja
# renderer, the GBNF compiler and the JSON parser recurse on HTTP and worker threads. std::thread
# takes the executable's reserve, so setting it on the link covers every thread.
string(APPEND CMAKE_EXE_LINKER_FLAGS " -Xlinker /STACK:8388608")

enable_language(RC)

# FILES OPEN IN BINARY MODE BY DEFAULT, as they do on Linux. binmode.obj sets the CRT's default
# translation mode; without it every fopen("r") and std::ifstream without ios::binary turns CRLF
# into LF and stops at a Ctrl-Z, so a chat template or tokenizer.json with Windows line endings
# would render differently than radiance does on Linux.
if(DEFINED ENV{VCToolsInstallDir} AND EXISTS "$ENV{VCToolsInstallDir}/lib/x64/binmode.obj")
  file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}/lib/x64/binmode.obj" BRL_BINMODE_OBJ)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " \"${BRL_BINMODE_OBJ}\"")
else()
  message(FATAL_ERROR "brilliance: binmode.obj not found under VCToolsInstallDir; configure from a "
                      "shell made by win\\env.ps1")
endif()

# ------------------------------------------------------------------------------- the POSIX layer
# win/compat: POSIX headers whose meaning carries over to Windows unchanged, so the radiance
# sources that use them compile as written (win/compat/include/brl_compat.h says what is in it and,
# more to the point, what is deliberately not). brl_prelude.h declares the POSIX functions that
# live in standard C headers on Linux and that the UCRT lacks; it is force-included so that no
# standard header has to be shadowed for them.
include_directories(BEFORE "${CMAKE_SOURCE_DIR}/win/compat/include")
add_compile_options("SHELL:-include \"${CMAKE_SOURCE_DIR}/win/compat/include/brl_prelude.h\"")
add_library(brl_compat STATIC
  "${CMAKE_SOURCE_DIR}/win/compat/src/compat_misc.cpp"
  "${CMAKE_SOURCE_DIR}/win/compat/src/compat_dl.cpp"
  "${CMAKE_SOURCE_DIR}/win/compat/src/compat_mman.cpp"
  "${CMAKE_SOURCE_DIR}/win/compat/src/compat_spawn.cpp")
# Every target links compiler-rt's builtins through it (BRL_CLANG_RT_BUILTINS, RadWindowsPlugin.cmake
# says why). Only in the tree: an installed compat library is linked with the consumer's own
# builtins, not with a path on the machine that built it.
target_link_libraries(brl_compat PUBLIC psapi ws2_32 "$<BUILD_INTERFACE:${BRL_CLANG_RT_BUILTINS}>")
# Defined before the top-level file sets the tree's C++20, so it says so itself.
target_compile_features(brl_compat PRIVATE cxx_std_20)
# Installed for plugins built against the installed radiance, as brilliance::compat: the library and
# its headers, which RadWindowsPlugin.cmake puts on their include path and force-includes as the
# tree's directories do here. The tree's own targets get the headers from those directories.
set_target_properties(brl_compat PROPERTIES EXPORT_NAME compat)
target_include_directories(brl_compat INTERFACE
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/brilliance/compat>)
install(TARGETS brl_compat EXPORT brillianceTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${CMAKE_SOURCE_DIR}/win/compat/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brilliance/compat)

# win/os: the services whose Windows meaning differs from POSIX, as explicit calls (win/os/os.h).
# A radiance source reaches them from an `#if defined(_WIN32)` branch as "os/os.h".
include_directories(AFTER "${CMAKE_SOURCE_DIR}/win")
add_library(brl_os STATIC "${CMAKE_SOURCE_DIR}/win/os/os_win32.cpp"
                          "${CMAKE_SOURCE_DIR}/win/os/gpu_win32.cpp"
                          "${CMAKE_SOURCE_DIR}/win/os/service_win32.cpp")
target_link_libraries(brl_os PUBLIC brl_compat advapi32 shell32 ole32 rstrtmgr psapi gdi32)
target_compile_features(brl_os PRIVATE cxx_std_20)
# Every target from here on -- the core, the tools, the tests, every plugin -- links both.
link_libraries(brl_compat brl_os)

# ------------------------------------------------------------------------------- HIP discovery
# The tree looks for hipcc in /opt/rocm and $ROCM_PATH and for the runtimes in their lib/, which
# the pinned SDK in .toolchain is not (win/env.ps1 leaves ROCM_PATH unset on purpose and sets
# HIP_PATH). The cache entries the tree's find_program/find_library fill are set here first, which
# keeps those calls as written: a variable already set is not searched again. Windows has no HSA
# runtime -- the AQL backend that writes HSA queues is Linux's and is left out of this build
# (brl_windows_finalize) -- so that entry names a target that links nothing.
if(DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" _brl_hip)
  if(EXISTS "${_brl_hip}/bin/hipcc.exe")
    set(RAD_HIPCC "${_brl_hip}/bin/hipcc.exe" CACHE FILEPATH "hipcc (win/cmake/RadWindows.cmake)")
  endif()
  if(EXISTS "${_brl_hip}/lib/amdhip64.lib")
    set(RAD_HIP_RUNTIME "${_brl_hip}/lib/amdhip64.lib" CACHE FILEPATH "the HIP runtime's import library")
  endif()
endif()
add_library(brl_no_hsa INTERFACE)
set(RAD_HSA_RUNTIME brl_no_hsa CACHE STRING "no HSA runtime on Windows (win/cmake/RadWindows.cmake)")

# The tests link `stdc++` ahead of rad_core (tests/CMakeLists.txt) to steer lld on Linux. There is
# no libstdc++ here -- the MSVC STL is in the CRT every module shares -- so the name is a target
# that links nothing, which keeps those lines as radiance wrote them.
add_library(stdc++ INTERFACE)

# ------------------------------------------------------------------------------- OpenMP
# Visual Studio's libomp, primed into FindOpenMP by the plugin half (RadWindowsPlugin.cmake says
# why); its omp.h goes ahead of MSVC's on every target's include path.
if(BRL_OMP_INCLUDE_DIR)
  include_directories(BEFORE SYSTEM "${BRL_OMP_INCLUDE_DIR}")
endif()

# The tree's few POSIX shell scripts (tests, tools/ops_doc_check.sh) run under the sh and awk Git
# for Windows installs; Git is already a prerequisite (win/setup.ps1).
file(TO_CMAKE_PATH "$ENV{ProgramFiles}" _brl_pf)
find_program(BRL_SH NAMES sh
             HINTS "${_brl_pf}/Git/usr/bin" "C:/Program Files/Git/usr/bin" "${_brl_pf}/Git/bin"
             NO_DEFAULT_PATH)

# ------------------------------------------------------------------------------- directory links
# `link` names `target` the way a POSIX build's symbolic link would. A symbolic link needs Developer
# Mode or an elevated process, and a network share may refuse one either way, so where the link
# cannot be made the directory is copied instead and every file in it becomes a configure
# dependency: an edit under `target` re-runs the configure, which refreshes the copy.
function(brl_link_dir target link)
  if(IS_SYMLINK "${link}")
    return()
  endif()
  if(NOT EXISTS "${link}")
    file(CREATE_LINK "${target}" "${link}" SYMBOLIC RESULT _rc)
    if(_rc EQUAL 0 AND EXISTS "${link}/")
      return()
    endif()
  endif()
  file(GLOB_RECURSE _files LIST_DIRECTORIES false RELATIVE "${target}" "${target}/*")
  foreach(f ${_files})
    configure_file("${target}/${f}" "${link}/${f}" COPYONLY)
  endforeach()
endfunction()

# ------------------------------------------------------------------------------- runtime DLLs
# Every executable lands in <build>/bin, and the DLLs it needs are copied there: Windows searches
# the exe's directory before System32, and System32 is where the AMD driver keeps its OWN
# amdhip64_7.dll. A HIP runtime found on PATH loses to that copy; one beside the exe wins.
set(BRL_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${BRL_BIN_DIR}")
function(brl_stage_runtime_dll path)
  if(EXISTS "${path}")
    file(COPY "${path}" DESTINATION "${BRL_BIN_DIR}")
    install(FILES "${path}" DESTINATION ${CMAKE_INSTALL_BINDIR})
    set_property(GLOBAL APPEND PROPERTY BRL_RUNTIME_DLLS "${path}")
  endif()
endfunction()
if(BRL_LIBOMP_DLL)
  brl_stage_runtime_dll("${BRL_LIBOMP_DLL}")
endif()

# ------------------------------------------------------------------------------- FFmpeg
# Upstream finds libavformat and its siblings through pkg-config (CMakeLists.txt, "media decoding")
# from the distribution. Windows has neither, so win/setup.ps1 installs the pinned LGPL shared build
# into .toolchain\ffmpeg and records it (toolchain.json ffmpeg_root, BRILLIANCE_FFMPEG_ROOT from
# win/env.ps1). This makes the target pkg_check_modules would have made, PkgConfig::RAD_FFMPEG,
# from that tree, and stages the five DLLs the engine links beside every executable and into the
# install, with FFmpeg's licence (dynamic linking keeps the LGPL's terms: the DLLs are replaceable).
# A macro: RAD_FFMPEG_FOUND and the version land in the caller's scope, as pkg_check_modules sets them.
macro(brl_find_ffmpeg)
  set(_brl_ff "$ENV{BRILLIANCE_FFMPEG_ROOT}")
  if(NOT _brl_ff)
    set(_brl_ff "${CMAKE_SOURCE_DIR}/.toolchain/ffmpeg")
  endif()
  file(TO_CMAKE_PATH "${_brl_ff}" _brl_ff)
  set(_brl_ff_libs avformat avcodec avutil swscale swresample)
  set(_brl_ff_ok ON)
  foreach(_l ${_brl_ff_libs})
    if(NOT EXISTS "${_brl_ff}/lib/${_l}.lib")
      set(_brl_ff_ok OFF)
    endif()
  endforeach()
  if(_brl_ff_ok AND EXISTS "${_brl_ff}/include/libavformat/avformat.h")
    if(NOT TARGET PkgConfig::RAD_FFMPEG)
      set(_brl_ff_implibs)
      foreach(_l ${_brl_ff_libs})
        list(APPEND _brl_ff_implibs "${_brl_ff}/lib/${_l}.lib")
      endforeach()
      add_library(PkgConfig::RAD_FFMPEG INTERFACE IMPORTED GLOBAL)
      set_target_properties(PkgConfig::RAD_FFMPEG PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_brl_ff}/include"
        INTERFACE_LINK_LIBRARIES "${_brl_ff_implibs}")
    endif()
    set(RAD_FFMPEG_FOUND ON)
    # The version upstream's status line prints, from libavformat's own headers.
    set(_brl_ff_v "")
    foreach(_part MAJOR MINOR MICRO)
      foreach(_h version_major.h version.h)
        if(EXISTS "${_brl_ff}/include/libavformat/${_h}")
          file(STRINGS "${_brl_ff}/include/libavformat/${_h}" _brl_ff_line
               REGEX "^#define LIBAVFORMAT_VERSION_${_part} +[0-9]+")
          if(_brl_ff_line)
            string(REGEX REPLACE ".* ([0-9]+).*" "\\1" _brl_ff_n "${_brl_ff_line}")
            string(APPEND _brl_ff_v "${_brl_ff_n}.")
            break()
          endif()
        endif()
      endforeach()
    endforeach()
    string(REGEX REPLACE "\\.$" "" RAD_FFMPEG_libavformat_VERSION "${_brl_ff_v}")
    foreach(_l ${_brl_ff_libs})
      file(GLOB _brl_ff_dll "${_brl_ff}/bin/${_l}-*.dll")
      foreach(_d ${_brl_ff_dll})
        brl_stage_runtime_dll("${_d}")
      endforeach()
    endforeach()
    if(EXISTS "${_brl_ff}/LICENSE.txt")
      install(FILES "${_brl_ff}/LICENSE.txt" DESTINATION share/doc/radiance/third_party
              RENAME ffmpeg.LICENSE.txt)
    endif()
  endif()
endmacro()

# ------------------------------------------------------------------------------- the host ABI
# On Linux an architecture plugin leaves rad_issue, rad_decl_* and the rest undefined and the
# dynamic linker binds them to whichever executable dlopen'ed it. A DLL must name the module every
# import comes from, so:
#
#   - every executable that hosts plugins EXPORTS the host ABI, from a .def file;
#   - every plugin imports it from a virtual module, brilliance_host.dll, DELAY-LOADED, and a hook
#     linked into the plugin answers the load of that module with the executable that is running.
#
# That reproduces the ELF semantics -- a plugin binds to radiance.exe, rad-convert.exe or a test,
# whichever loaded it -- with no change to the ABI and none to plugin sources.
#
# The list is not written down: it is every function DECLARED in the headers that hold the host's
# half of the ABI, which is the definition of that half. The headers are configure dependencies, so
# a function added there reaches the export list on the next configure.
set(BRL_HOST_ABI_HEADERS
  ${CMAKE_SOURCE_DIR}/abi/rad_builder.h
  ${CMAKE_SOURCE_DIR}/abi/rad_runtime.h
  ${CMAKE_SOURCE_DIR}/abi/rad_device.h)
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${BRL_HOST_ABI_HEADERS})
set(BRL_HOST_FUNCS "")
foreach(h ${BRL_HOST_ABI_HEADERS})
  file(STRINGS "${h}" _lines REGEX "^[A-Za-z_][A-Za-z0-9_ ]*[ *]+rad_[a-z0-9_]+[ \t]*\\(")
  foreach(l ${_lines})
    if(l MATCHES "^(static|typedef|#)")
      continue()
    endif()
    string(REGEX MATCH "rad_[a-z0-9_]+[ \t]*\\(" _fn "${l}")
    string(REGEX REPLACE "[ \t]*\\($" "" _fn "${_fn}")
    # rad_arch_* are declared beside the builder API but EXPORTED BY the architecture plugin.
    if(_fn MATCHES "^rad_arch_")
      continue()
    endif()
    list(APPEND BRL_HOST_FUNCS ${_fn})
  endforeach()
endforeach()
list(REMOVE_DUPLICATES BRL_HOST_FUNCS)
list(SORT BRL_HOST_FUNCS)
list(LENGTH BRL_HOST_FUNCS _n)
if(_n LESS 40)
  message(FATAL_ERROR "brilliance: read only ${_n} host ABI functions out of abi/ -- the header "
                      "parser in win/cmake/RadWindows.cmake no longer matches the headers")
endif()
message(STATUS "brilliance: host ABI ${_n} functions")

set(_exports "EXPORTS\n")
foreach(f ${BRL_HOST_FUNCS})
  string(APPEND _exports "  ${f}\n")
endforeach()
set(BRL_HOST_EXPORTS_DEF "${CMAKE_BINARY_DIR}/brilliance_host_exports.def")
file(CONFIGURE OUTPUT "${BRL_HOST_EXPORTS_DEF}" CONTENT "${_exports}" @ONLY)
set(BRL_HOST_IMPORT_DEF "${CMAKE_BINARY_DIR}/brilliance_host.def")
file(CONFIGURE OUTPUT "${BRL_HOST_IMPORT_DEF}" CONTENT "LIBRARY brilliance_host.dll\n${_exports}" @ONLY)

# The import library plugins link, made from the .def at configure time: it changes only when the
# headers do, and those re-run the configure.
get_filename_component(_cxxdir "${CMAKE_CXX_COMPILER}" DIRECTORY)
find_program(BRL_LIB_TOOL NAMES llvm-lib lib HINTS "${_cxxdir}")
if(NOT BRL_LIB_TOOL)
  message(FATAL_ERROR "brilliance: neither llvm-lib nor lib.exe found to make the host import library")
endif()
set(BRL_HOST_IMPLIB "${CMAKE_BINARY_DIR}/brilliance_host.lib")
execute_process(
  COMMAND "${BRL_LIB_TOOL}" /nologo /machine:x64 "/def:${BRL_HOST_IMPORT_DEF}" "/out:${BRL_HOST_IMPLIB}"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "brilliance: making ${BRL_HOST_IMPLIB} failed:\n${_out}${_err}")
endif()

# What a plugin links to reach the host (brl_win_plugin, in RadWindowsPlugin.cmake). An INTERFACE
# target, so the hook source, the import library and the delay-load flags travel together to every
# plugin in one link line. Installed as brilliance::host_imports, with the import library and the
# hook source, for plugins built against the installed radiance; brilliance::host_imports names it
# in both builds.
add_library(brl_host_imports INTERFACE)
add_library(brilliance::host_imports ALIAS brl_host_imports)
set_target_properties(brl_host_imports PROPERTIES EXPORT_NAME host_imports)
target_sources(brl_host_imports INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/win/cmake/rad_delayload_hook.cpp>"
  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/cmake/radiance/rad_delayload_hook.cpp>")
target_link_libraries(brl_host_imports INTERFACE
  "$<BUILD_INTERFACE:${BRL_HOST_IMPLIB}>"
  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/brilliance_host.lib>"
  delayimp.lib)
target_link_options(brl_host_imports INTERFACE "LINKER:/DELAYLOAD:brilliance_host.dll")
install(TARGETS brl_host_imports EXPORT brillianceTargets)
install(FILES "${BRL_HOST_IMPLIB}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
# brilliance::compat and brilliance::host_imports, which the installed RadWindowsPlugin.cmake reads.
install(EXPORT brillianceTargets
        NAMESPACE brilliance::
        FILE brillianceTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radiance)

# An executable that loads architecture plugins presents the host ABI to them. Called wherever the
# Linux build sets ENABLE_EXPORTS; WHOLE_ARCHIVE there already pulls every definition in.
function(brl_win_host_exports target)
  target_link_options(${target} PRIVATE "LINKER:/DEF:${BRL_HOST_EXPORTS_DEF}")
  # A link map beside the executable: the build has no PDB, and a hang's stack (the engine's step
  # watchdog, win/tools/stacks/brl-stacks) prints exe+offset, which the map turns into functions.
  target_link_options(${target} PRIVATE "LINKER:/MAP:$<TARGET_FILE_DIR:${target}>/${target}.map")
endfunction()

# Every plugin gets brl_win_plugin() (RadWindowsPlugin.cmake) from rad_add_plugin.

# ------------------------------------------------------------------------------- finishing touches
# Run at the end of the top-level CMakeLists, when every target exists: every executable gets the
# manifest (win/brilliance.manifest: UTF-8 as its active code page, so the narrow CRT and
# std::filesystem take the UTF-8 paths the tree uses everywhere; long paths; Windows 10+), every
# executable that hosts plugins exports the host ABI, and each lands in <build>/bin beside the
# runtime DLLs unless it said otherwise.
function(_brl_collect_targets dir out)
  get_property(_t DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(_subs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(s ${_subs})
    _brl_collect_targets("${s}" _more)
    list(APPEND _t ${_more})
  endforeach()
  set(${out} ${_t} PARENT_SCOPE)
endfunction()

function(brl_windows_finalize)
  # THE DEVICE BACKEND. core/CMakeLists.txt globs core/device, which brings the AQL backend: HSA
  # queues the engine writes itself, the fat-binary registration it interposes (aql_code.cpp) and
  # its copy kernels (aql_blit.hip). None of that exists on Windows, where the HIP runtime owns the
  # queues; the native HIP backend (win/device/hipnative.cpp) takes its place, and the Linux-only
  # link options that served AQL -- an rpath, and the HIP entry points it re-exports for plugins to
  # bind to -- are dropped. Done here, on the finished target, so the core's CMakeLists stays as
  # radiance wrote it.
  # Only in a HIP build: without HIP, aql.cpp compiles to the stubs the host build and its tests link
  # (ride_reusable and the rest), and aql_code.cpp -- code-object parsing, ELF whatever the host -- is
  # portable and stays in every build.
  if(TARGET rad_core AND RAD_HIP_FOUND)
    get_target_property(_src rad_core SOURCES)
    list(FILTER _src EXCLUDE REGEX "/core/device/(aql\\.cpp|aql_blit\\.hip)$")
    list(APPEND _src "${CMAKE_SOURCE_DIR}/win/device/hipnative.cpp")
    set_property(TARGET rad_core PROPERTY SOURCES "${_src}")
    get_target_property(_lo rad_core INTERFACE_LINK_OPTIONS)
    if(_lo)
      list(FILTER _lo EXCLUDE REGEX "^LINKER:(-rpath,|--export-dynamic-symbol=)")
      set_property(TARGET rad_core PROPERTY INTERFACE_LINK_OPTIONS "${_lo}")
    endif()
  endif()

  # THE HUGGING FACE FETCHER. core/hf_model.cpp runs <exe>\..\libexec\brilliance\hfget.ps1 for a
  # --model that is a repo id; the build tree and the install both put it there (hfget.sh, which
  # needs sh, curl and jq, stays in libexec/radiance for Linux).
  configure_file("${CMAKE_SOURCE_DIR}/win/libexec/hfget.ps1"
                 "${CMAKE_BINARY_DIR}/libexec/brilliance/hfget.ps1" COPYONLY)
  install(FILES "${CMAKE_SOURCE_DIR}/win/libexec/hfget.ps1" DESTINATION libexec/brilliance)

  # THE GPU TELEMETRY's own test (win/os/gpu_win32.cpp): reads adapter data only, starts no device
  # work, holds trivially on a machine without a hardware adapter.
  if(RAD_BUILD_TESTS)
    add_executable(os_gpu_test "${CMAKE_SOURCE_DIR}/win/os/os_gpu_test.cpp")
    target_include_directories(os_gpu_test PRIVATE "${CMAKE_SOURCE_DIR}/win/os")
    set_target_properties(os_gpu_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME os_gpu_test COMMAND os_gpu_test)
  endif()

  # A HUNG PROCESS'S STACKS, from outside it (win/tools/stacks/README.md): every thread, each held
  # only for the moment its stack is read. Plain Win32, no HIP; in every build and the package.
  add_executable(brl-stacks "${CMAKE_SOURCE_DIR}/win/tools/stacks/brl-stacks.cpp")
  set_target_properties(brl-stacks PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  install(TARGETS brl-stacks RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES "${CMAKE_SOURCE_DIR}/win/tools/stacks/resolve-stacks.py"
          DESTINATION libexec/brilliance)

  # THE CONVERSION KIT. Upstream's image copies data/recipes into share/radiance/recipes (Dockerfile),
  # where rad-export and rad-convert's users find them beside the installed tools; the install here
  # does the same, and gives rad-export (a Python script, installed by upstream's rule) a .cmd that
  # runs it with the machine's Python.
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/data/recipes/" DESTINATION share/radiance/recipes
          FILES_MATCHING PATTERN "*.recipe")
  install(PROGRAMS "${CMAKE_SOURCE_DIR}/win/libexec/rad-export.cmd" DESTINATION ${CMAKE_INSTALL_BINDIR})

  # THE MACHINE'S FACTS FROM WINDOWS. core/diag/sysinfo.cpp reads /proc and /sys; on the live
  # machine a Windows build asks Win32 instead (win/diag/sysinfo_win32.cpp), in every build.
  if(TARGET rad_core)
    target_sources(rad_core PRIVATE "${CMAKE_SOURCE_DIR}/win/diag/sysinfo_win32.cpp")
    # The collective trace core/runtime/issue.cpp feeds and a stall prints (win/device/coll_trace.h).
    target_sources(rad_core PRIVATE "${CMAKE_SOURCE_DIR}/win/device/coll_trace.cpp")
  endif()

  # BRILLIANCE'S OWN IDENTITY, beside upstream's version and plugin ABI: the release, the source
  # commit the binaries come from (git asked at every build, so it never goes stale) and the
  # upstream pin. /version, /props and the startup report say it (core/server/admin.cpp,
  # core/diag/report.cpp, their _WIN32 branches), and a generated video's metadata names the
  # release (core/server/videos.cpp). Only those three files see the generated header, so a new
  # commit recompiles them and nothing else.
  if(TARGET rad_core)
    set(_id_dir "${CMAKE_BINARY_DIR}/brl")
    add_custom_target(brl_identity ALL
      COMMAND "${CMAKE_COMMAND}" "-DSRC=${CMAKE_SOURCE_DIR}" "-DOUT=${_id_dir}/brl_identity.h"
              "-DVERSION=${BRILLIANCE_VERSION}" "-DUPSTREAM=${BRILLIANCE_UPSTREAM}"
              -P "${CMAKE_SOURCE_DIR}/win/cmake/brl_identity.cmake"
      BYPRODUCTS "${_id_dir}/brl_identity.h"
      COMMENT "Brilliance identity (release, source commit, upstream pin)"
      VERBATIM)
    add_dependencies(rad_core brl_identity)
    set_source_files_properties("${CMAKE_SOURCE_DIR}/core/server/admin.cpp"
                                "${CMAKE_SOURCE_DIR}/core/diag/report.cpp"
                                "${CMAKE_SOURCE_DIR}/core/server/videos.cpp"
                                TARGET_DIRECTORY rad_core
                                PROPERTIES INCLUDE_DIRECTORIES "${_id_dir}")
  endif()

  # THE HIP RUNTIME THE TREE WAS BUILT AGAINST, beside every executable and in the install: the
  # SDK's bin\ next to the lib\ the import library came from. Without these the exe loads the
  # driver's own amdhip64_7.dll from System32, a different HIP than the kernels were compiled for.
  if(RAD_HIP_FOUND AND RAD_HIP_RUNTIME)
    get_filename_component(_hip_lib "${RAD_HIP_RUNTIME}" DIRECTORY)
    get_filename_component(_hip_bin "${_hip_lib}/../bin" ABSOLUTE)
    foreach(_dll amdhip64_7.dll amd_comgr.dll rocm_kpack.dll)
      if(EXISTS "${_hip_bin}/${_dll}")
        brl_stage_runtime_dll("${_hip_bin}/${_dll}")
      else()
        message(WARNING "brilliance: ${_dll} not found in ${_hip_bin}; the HIP runtime will come from System32")
      endif()
    endforeach()
    # amd_comgr's licence, from the same SDK. The SDK wheel carries none for amdhip64_7 (HIP's CLR)
    # or rocm_kpack (found in the licence check).
    get_filename_component(_comgr_lic "${_hip_lib}/../share/doc/amd_comgr/LICENSE.txt" ABSOLUTE)
    if(EXISTS "${_comgr_lic}")
      install(FILES "${_comgr_lic}" DESTINATION share/doc/radiance/third_party RENAME amd_comgr.LICENSE.txt)
    endif()
  endif()

  # THE LICENCES OF WHAT THE WINDOWS BUILD ADDS (win/licenses): LLVM's for libomp.dll, the terms the
  # VC++ runtime is distributed under, and where the FFmpeg DLLs' source is.
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/win/licenses/" DESTINATION share/doc/radiance/third_party)
  # THE WINDOWS CONTRACT TESTS (win/tests), beside the tree's own; before the pass below, so their
  # executables get the manifest and the runtime DLLs too.
  if(RAD_BUILD_TESTS AND TARGET radiance)
    add_subdirectory("${CMAKE_SOURCE_DIR}/win/tests" "${CMAKE_BINARY_DIR}/win/tests")
  endif()

  _brl_collect_targets("${CMAKE_SOURCE_DIR}" _all)
  set(_dirs "${BRL_BIN_DIR}")
  foreach(t ${_all})
    get_target_property(_type ${t} TYPE)
    # A MODULE library is a plugin whether or not rad_add_plugin made it (the tests build one with
    # add_library directly), and a plugin is found by its exports: ELF exports everything, so here
    # every module does too.
    if(_type STREQUAL "MODULE_LIBRARY")
      set_target_properties(${t} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    if(_type STREQUAL "EXECUTABLE")
      target_sources(${t} PRIVATE "${CMAKE_SOURCE_DIR}/win/brilliance.manifest")
      # ENABLE_EXPORTS is how the Linux build marks an executable whose symbols plugins bind to.
      get_target_property(_exp ${t} ENABLE_EXPORTS)
      if(_exp)
        brl_win_host_exports(${t})
      endif()
      get_target_property(_rod ${t} RUNTIME_OUTPUT_DIRECTORY)
      # An executable radiance keeps INSIDE a plugin directory would put its runtime DLLs there
      # too, where the plugin scan takes every .dll for a plugin. On Windows those go to bin/.
      if(_rod AND RAD_PLUGIN_OUT)
        string(FIND "${_rod}" "${RAD_PLUGIN_OUT}" _in_home)
        if(_in_home EQUAL 0)
          set(_rod "")
        endif()
      endif()
      if(NOT _rod)
        set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${BRL_BIN_DIR}")
      else()
        list(APPEND _dirs "${_rod}")
      endif()
    endif()
  endforeach()
  # An executable the build tree keeps outside bin/ (the tests) loads plugins whose own
  # dependencies -- libomp, the HIP runtime -- are resolved from the EXECUTABLE's directory. The
  # installed tree has every executable in bin/; the build tree gets copies.
  list(REMOVE_DUPLICATES _dirs)
  get_property(_dlls GLOBAL PROPERTY BRL_RUNTIME_DLLS)
  foreach(d ${_dirs})
    file(MAKE_DIRECTORY "${d}")
    foreach(dll ${_dlls})
      file(COPY "${dll}" DESTINATION "${d}")
    endforeach()
  endforeach()
endfunction()
