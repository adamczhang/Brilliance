# Brilliance: x64 Windows, MSVC, dynamic CRT (/MD), release only.
# Every dependency is a static library; FFmpeg alone is shared, so the result is the five FFmpeg
# DLLs (avcodec, avformat, avutil, swscale, swresample) with dav1d, libwebp and zlib linked inside
# (and AMD's AMF headers compiled in; they are headers only).
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_BUILD_TYPE release)

if(PORT STREQUAL "ffmpeg")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

# The repository's pinned compiler (.toolchain\toolchain.json): Visual Studio Build Tools 18,
# MSVC 14.51. build-ffmpeg.ps1 passes the instance path through BRL_VS_PATH.
if(DEFINED ENV{BRL_VS_PATH})
    set(VCPKG_VISUAL_STUDIO_PATH "$ENV{BRL_VS_PATH}")
endif()
set(VCPKG_PLATFORM_TOOLSET_VERSION 14.51)

# The build environment is otherwise cleaned by vcpkg; let the pinned instance path through.
set(VCPKG_ENV_PASSTHROUGH BRL_VS_PATH)
