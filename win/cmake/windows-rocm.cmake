# Brilliance toolchain: ROCm's clang for C, C++ and HIP on Windows, against the MSVC STL/UCRT.
#
# Use from a shell made by win\env.ps1 (which imports the pinned vcvars64 and sets HIP_PATH to the
# project's own ROCm in .toolchain\), or let win\build.ps1 do both:
#   cmake -S . -B build-win\host -G Ninja -DCMAKE_TOOLCHAIN_FILE=win/cmake/windows-rocm.cmake
#
# One compiler for every language is not a preference. On Windows only a unified HIP compiler is
# supported, and the tree's ABI headers use C compound literals in C++ that MSVC's cl.exe rejects.

if(NOT DEFINED ENV{HIP_PATH})
  message(FATAL_ERROR "HIP_PATH is not set: run this configure from a shell made by win\\env.ps1")
endif()
file(TO_CMAKE_PATH "$ENV{HIP_PATH}" BRL_ROCM_ROOT)

set(CMAKE_C_COMPILER   "${BRL_ROCM_ROOT}/lib/llvm/bin/clang.exe")
set(CMAKE_CXX_COMPILER "${BRL_ROCM_ROOT}/lib/llvm/bin/clang++.exe")
set(CMAKE_HIP_COMPILER "${BRL_ROCM_ROOT}/lib/llvm/bin/clang++.exe")
set(CMAKE_HIP_PLATFORM amd)
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${BRL_ROCM_ROOT}")
# hip-config-amd.cmake's own architecture detection looks for a layout TheRock does not have;
# naming the targets keeps it from guessing.
set(GPU_TARGETS "gfx1201" CACHE STRING "")
list(PREPEND CMAKE_PREFIX_PATH "${BRL_ROCM_ROOT}")
