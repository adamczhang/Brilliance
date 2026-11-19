# Brilliance: the plugin half of the Windows build -- what every plugin needs on Windows, built in
# radiance's tree or out of it against an installed radiance.
#
# win/cmake/RadWindows.cmake includes this file for the tree's own build. `cmake --install` puts it
# in <prefix>/lib/cmake/radiance beside RadAddPlugin.cmake, and the installed radiance-config.cmake
# includes it on Windows, so a plugin configured with find_package(radiance) compiles and links the
# way the tree's own plugins do. One copy for both, for the reason cmake/RadAddPlugin.cmake gives:
# a drifted duplicate builds plugins that compile and then do not load.
#
# For both builds it provides:
#   - the compiler check: ROCm's clang++ through the GNU driver;
#   - the compile definitions and warning flags every Windows translation unit takes
#     (BRL_WIN_DEFINITIONS, BRL_WIN_CXX_OPTIONS) and the shared CRT;
#   - compiler-rt's builtins (BRL_CLANG_RT_BUILTINS);
#   - FindOpenMP primed with Visual Studio's libomp, and the directory holding its omp.h
#     (BRL_OMP_INCLUDE_DIR);
#   - brl_win_plugin(), which rad_add_plugin calls for every plugin.
#
# Out of the tree it then imports what the tree built for plugins and installed (brillianceTargets:
# brilliance::compat, the POSIX layer, and brilliance::host_imports, the host ABI's import library
# with its delay-load hook), and puts on radiance::abi what the tree applies to every target in its
# directories: the definitions and flags above, the force-included prelude, the compat headers and
# library, and LLVM's omp.h. Every source of a plugin links radiance::abi, so it compiles as in the
# tree, and nothing else in the consumer's project changes. The executables a plugin project may
# build beside its plugins (libavx's harnesses) get the same, but not the engine's executable
# settings (manifest, stack size, binary stdio), which belong to the programs that host plugins.

include_guard(GLOBAL)

# ------------------------------------------------------------------------------- the compiler
# ROCm's clang++ and nothing else. MSVC cl.exe cannot compile the ABI's C compound literals in
# C++ (RAD_OPD, RAD_PARAMS, RAD_ISSUE: thousands of sites), and clang-cl would take every
# GNU-style flag in the tree away from us. The GNU driver targeting x86_64-pc-windows-msvc keeps
# the tree's flags and links against the MSVC STL and UCRT, which is what the HIP runtime's
# headers and DLLs are built for.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  message(FATAL_ERROR
    "Brilliance builds with ROCm's clang++ (GNU driver), not '${CMAKE_CXX_COMPILER_ID}' with the "
    "'${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}' front end. Configure from a shell made by "
    "win\\env.ps1 with -DCMAKE_TOOLCHAIN_FILE=win/cmake/windows-rocm.cmake, or use win\\build.ps1; "
    "a plugin built against an installed radiance passes that clang++ as -DCMAKE_CXX_COMPILER.")
endif()

# _WIN32_WINNT 0x0A00: Windows 10+ APIs (PrefetchVirtualMemory, VirtualAlloc2, high-resolution
# waitable timers). WIN32_LEAN_AND_MEAN and NOMINMAX because <windows.h> otherwise defines min/max
# macros that break std::min across the tree. _USE_MATH_DEFINES for M_PI, which the UCRT hides.
set(BRL_WIN_DEFINITIONS
  _WIN32_WINNT=0x0A00 WINVER=0x0A00 WIN32_LEAN_AND_MEAN NOMINMAX
  _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS _USE_MATH_DEFINES)

# The MSVC STL's headers trip warnings clang finds pedantic; ours are what we read.
set(BRL_WIN_CXX_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wno-language-extension-token>)

# ONE CRT FOR EVERY MODULE (/MD). The engine sets environment variables that plugins read with
# getenv, memory crosses module boundaries, and plugins are loaded into whichever exe hosts them.
# With a static CRT each DLL has its own heap and its own copy of the environment.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# COMPILER-RT BUILTINS, which the GNU driver links on Linux and not for an MSVC target.
# __builtin_cpu_supports -- libavx's and libquant's ISA dispatch -- reads __cpu_model, which lives
# there (and is filled in by compiler-rt's own initialiser), as do the 128-bit division helpers.
# Found beside the compiler of the build at hand, so an installed radiance names no path of the
# machine it was built on.
get_filename_component(_brl_llvm_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
file(GLOB BRL_CLANG_RT_BUILTINS
     "${_brl_llvm_bin}/../lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
if(NOT BRL_CLANG_RT_BUILTINS)
  message(FATAL_ERROR "brilliance: clang_rt.builtins-x86_64.lib not found beside ${CMAKE_CXX_COMPILER}")
endif()
list(GET BRL_CLANG_RT_BUILTINS 0 BRL_CLANG_RT_BUILTINS)

# ------------------------------------------------------------------------------- OpenMP
# libavx, libquant, libref and rad-convert use OpenMP beyond MSVC's vcomp (collapse, simd). ROCm's
# Windows LLVM ships no OpenMP runtime; Visual Studio's LLVM component does, and it is the same
# libomp ABI clang's -fopenmp emits calls for. FindOpenMP is primed with it so the tree's own
# find_package(OpenMP) calls succeed unchanged.
set(BRL_LIBOMP_DIR "$ENV{BRILLIANCE_LIBOMP_DIR}" CACHE PATH "directory holding lib/libomp.lib and bin/libomp.dll (VS LLVM x64)")
if(NOT BRL_LIBOMP_DIR)
  foreach(_vc "$ENV{VCINSTALLDIR}" "$ENV{VSINSTALLDIR}/VC")
    if(_vc AND EXISTS "${_vc}/Tools/Llvm/x64/lib/libomp.lib")
      file(TO_CMAKE_PATH "${_vc}/Tools/Llvm/x64" BRL_LIBOMP_DIR)
      set(BRL_LIBOMP_DIR "${BRL_LIBOMP_DIR}" CACHE PATH "" FORCE)
      break()
    endif()
  endforeach()
endif()
file(TO_CMAKE_PATH "${BRL_LIBOMP_DIR}" BRL_LIBOMP_DIR)
set(BRL_OMP_INCLUDE_DIR "")
if(BRL_LIBOMP_DIR AND EXISTS "${BRL_LIBOMP_DIR}/lib/libomp.lib")
  set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "")
  set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "")
  set(OpenMP_C_LIB_NAMES "libomp" CACHE STRING "")
  set(OpenMP_CXX_LIB_NAMES "libomp" CACHE STRING "")
  set(OpenMP_libomp_LIBRARY "${BRL_LIBOMP_DIR}/lib/libomp.lib" CACHE FILEPATH "")
  set(BRL_LIBOMP_DLL "${BRL_LIBOMP_DIR}/bin/libomp.dll")
  # LLVM's omp.h, and only it, ahead of the MSVC one: MSVC's describes vcomp's OpenMP 2.0 and lacks
  # what the tree calls (omp_set_max_active_levels). Copied into a directory of its own because
  # the directory it comes from is a whole, older clang resource directory that must not shadow
  # ROCm clang's builtin headers. The includer puts BRL_OMP_INCLUDE_DIR first on the include path.
  file(GLOB _brl_omp_h "${BRL_LIBOMP_DIR}/lib/clang/*/include/omp.h")
  if(_brl_omp_h)
    list(GET _brl_omp_h 0 _brl_omp_h)
    file(COPY "${_brl_omp_h}" DESTINATION "${CMAKE_BINARY_DIR}/brl_omp_include")
    set(BRL_OMP_INCLUDE_DIR "${CMAKE_BINARY_DIR}/brl_omp_include")
  endif()
  message(STATUS "brilliance: OpenMP from ${BRL_LIBOMP_DIR}")
else()
  message(WARNING "brilliance: no libomp found (install the VS 'C++ Clang tools' component); "
                  "host kernels will run single-threaded")
  # Left to itself FindOpenMP finds clang's -fopenmp with MSVC's omp.h (vcomp's OpenMP 2.0), which
  # does not compile the tree. Every find_package(OpenMP) in the tree is QUIET, so this is the
  # serial build the warning promises.
  set(CMAKE_DISABLE_FIND_PACKAGE_OpenMP TRUE)
endif()

# ------------------------------------------------------------------------------- every plugin
# Called by rad_add_plugin (cmake/RadAddPlugin.cmake).
#
# EXPORTS: every external symbol, generated by CMake from the objects. ELF exports a plugin's
# functions by default and the loader resolves them by name -- the fixed entry points, the
# OPTIONAL ones the ABI only names as typedefs (rad_plugin_open, rad_kernel_concurrent, ...), and
# the test hooks harnesses look up. Exporting everything keeps all three without touching a plugin
# source, which is the property that keeps radiance's plugins mergeable.
#
# IMPORTS: the host ABI from the virtual brilliance_host.dll, delay-loaded and bound by the hook to
# whichever executable loaded the plugin (brilliance::host_imports; RadWindows.cmake says why).
function(brl_win_plugin target)
  set_target_properties(${target} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  target_link_libraries(${target} PRIVATE brilliance::host_imports)
endfunction()

# ------------------------------------------------------------------------------- out of the tree
# The tree makes brilliance::host_imports and the compat library itself and applies the rest to its
# directories (RadWindows.cmake); it stops here.
if(BRL_IN_TREE)
  return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/brillianceTargets.cmake")
# The tree links the builtins of the compiler it was built with; a consumer links its own.
set_property(TARGET brilliance::compat APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${BRL_CLANG_RT_BUILTINS}")

# What the tree gives every target in its directories, as usage requirements of the ABI target.
# brl_prelude.h is force-included (win/compat/include/brl_prelude.h says why); the joined
# -include<file> spelling keeps a path with a space in it one argument.
get_target_property(_brl_compat_inc brilliance::compat INTERFACE_INCLUDE_DIRECTORIES)
set_property(TARGET radiance::abi APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${BRL_WIN_DEFINITIONS})
set_property(TARGET radiance::abi APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
             ${BRL_WIN_CXX_OPTIONS} "-include${_brl_compat_inc}/brl_prelude.h")
if(BRL_OMP_INCLUDE_DIR)
  set_property(TARGET radiance::abi APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${BRL_OMP_INCLUDE_DIR}")
  set_property(TARGET radiance::abi APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${BRL_OMP_INCLUDE_DIR}")
endif()
set_property(TARGET radiance::abi APPEND PROPERTY INTERFACE_LINK_LIBRARIES brilliance::compat)
