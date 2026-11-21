# Third-party software in the Brilliance Windows package

This is the index of the third-party DLLs in the package's `bin\` and of the libraries built into
them, with the licence each is used under and the file that carries its text.

- **Where the files are.** In the package: `share\doc\radiance\third_party`, beside this file. In
  the source tree: `win\licenses`. The build adds two more texts from what it was built against:
  `ffmpeg.LICENSE.txt` (FFmpeg's LGPL-2.1 text, the FFmpeg zip's `LICENSE.txt`) and
  `amd_comgr.LICENSE.txt` (from the ROCm SDK).
- **What is not here.** The engine's own third-party code (cpp-httplib, yyjson, llama.cpp and the
  rest) is named in `NOTICE`, one directory up, and its texts are beside this file too.
- **The texts are upstream's, unedited**, taken at the commit or tag the table gives. Two files
  are extracts: `elfio.NOTICE` is the licence comment at the top of an upstream header (the tree
  has no licence file there), and `llvm-openmp.LICENSE` is LLVM's entry in Visual Studio's
  ThirdPartyNotices. Each file keeps upstream's bytes, so a text that refers to `COPYING` or
  `LICENSES/...` means the file the table lists under that upstream name.
- **Collected 2026-11-17**; the FFmpeg rows 2026-11-18, from the build of Brilliance's own
  FFmpeg (`win\ffmpeg\THIRD-PARTY-ffmpeg.md`). The last section records where each text was
  fetched and its SHA-256.

## The DLLs

| DLLs in `bin\` | What they are | Built from | Licence | Files |
|---|---|---|---|---|
| `amdhip64_7.dll` | AMD's HIP runtime (HIP 7.16.26385) | AMD's `rocm_sdk_core` 10.1.0 wheel: TheRock `b324869fb3116353ac7eec5e6926aa3d452ac987`, rocm-systems `7da43026e4f2ef7600433c38c820732f7a205d22` | MIT and NCSA, with AMD's End User License Agreement for the directory PAL comes from; see "Inside amdhip64_7.dll" and "AMD's EULA in amdhip64_7.dll" | listed below |
| `rocm_kpack.dll` | ROCm's kernel-pack reader | the same wheel and commits (rocm-systems `shared/kpack`) | MIT, with BSL-1.0 and BSD-3-Clause parts | listed below |
| `amd_comgr.dll` | AMD's code-object manager | the same wheel | Apache-2.0 WITH LLVM-exception | `amd_comgr.LICENSE.txt` |
| `avcodec-62.dll`, `avformat-62.dll`, `avutil-60.dll`, `swresample-6.dll`, `swscale-9.dll` | FFmpeg 8.1.3, Brilliance's own build: decoding, the picture encoders, and H.264 only through the `h264_amf` and `h264_mf` wrappers (see "H.264") | `n8.1.3` (https://github.com/ffmpeg/ffmpeg/archive/n8.1.3.tar.gz) with vcpkg's `ffmpeg` port patches, built by `win\ffmpeg` (vcpkg registry tag 2026.07.29, MSVC 14.51.36231): `brilliance-ffmpeg-8.1.3-win64-lgpl-20261118.zip` | LGPL-2.1-or-later (configured without `--enable-gpl`, `--enable-nonfree` or `--enable-version3`; each DLL reports "LGPL version 2.1 or later") | `ffmpeg.LICENSE.txt`, `ffmpeg.SOURCE`, and the libraries below |
| `libomp.dll` | LLVM's OpenMP runtime | Visual Studio Build Tools 18 (`VC\Tools\Llvm\x64\bin`) | Apache-2.0 WITH LLVM-exception AND NCSA | `llvm-openmp.LICENSE` |
| `vcruntime140*.dll`, `msvcp140*.dll`, `concrt140.dll`, `vccorlib140.dll` | Microsoft's Visual C++ runtime | Visual Studio Build Tools 18 (`Microsoft.VC145.CRT`) | Visual Studio licence terms, Distributable Code | `msvc-runtime.NOTICE` |

The FFmpeg DLLs link no GCC, MinGW or winpthreads runtime. They import Windows system DLLs, the
Universal CRT and the VC++ runtime (`msvc-runtime.NOTICE`). Three more DLLs are loaded at run
time, and the package ships none of them: Media Foundation's `mfplat.dll` and `d3d11.dll`, which
come with Windows, and AMD's AMF runtime, `amfrt64.dll`, from the AMD driver. FFmpeg loads them
for the media paths built to use them: the two H.264 encoders and the AMF and D3D11 device
contexts those encoders open. Hardware decoding is configured out (`--disable-hwaccels`).

## H.264

What ships and what does not:

- **No H.264 encoder implementation ships in Brilliance 0.3.0.** OpenH264 and x264 are not in
  the package; `avcodec-62.dll` holds none of OpenH264's `Wels*` symbols.
- **The bundled FFmpeg keeps exactly two H.264 encoders.** Both are thin wrappers that hand the
  work to an encoder its maker licenses; neither contains an H.264 encoder of its own.
  - `h264_amf`: AMD's hardware encoder, through AMF. The AMF runtime comes with the AMD driver,
    not with this package.
  - `h264_mf`: Windows' Media Foundation encoder. Windows "N" editions lack it unless Microsoft's
    Media Feature Pack is installed.
  - H.264 output goes only through these two, AMD's first by default (`core/mm/mp4.h`).
    `BRILLIANCE_H264_ENCODER` can restrict the choice to one of them or change their order, and
    refuses any other name; the engine takes no other H.264 encoder, even from a different FFmpeg.
- **What produces H.264.** Video generation (`/v1/videos`) writes its MP4s through them, and an
  image-to-video model's first frame goes through an H.264 round trip with the same two. Video
  generation works on one card and is not yet qualified. Speech synthesis (TTS) is not supported
  in 0.3.0, and music generation, which shares its route, is not served; music writes WAV or raw
  PCM itself and needs no encoder.
- **Who covers H.264 output.** H.264 output through `h264_amf` or `h264_mf` is covered by the
  licence terms of that encoder's vendor (AMD for AMF, Microsoft for Media Foundation), not by
  anything this project supplies. This describes how the package is put together; it is not legal
  advice from the project.
- **Why no H.264 encoder ships inside.** Cisco's royalty-free H.264 patent licence covers only
  the OpenH264 binaries Cisco builds and distributes, not OpenH264 compiled into another DLL.
  AMD's and Microsoft's encoders come licensed from their
  makers, so H.264 output goes only through them.
- **Decoding H.264 input** (video sent in requests) uses FFmpeg's built-in H.264 decoder. H.264
  decoding is patent-encumbered, and the package supplies no H.264 patent licence.
- **The source of 0.3.0's FFmpeg.** It is built with vcpkg from the recipe in the source tree's
  `win\ffmpeg\`, as a zip pinned by its SHA-256 (`win\setup.ps1`). The zip and FFmpeg's own
  `n8.1.3` source archive are assets of the 0.3.0-alpha.1 release
  (https://github.com/adamczhang/Brilliance/releases/tag/v0.3.0-alpha.1); `ffmpeg.SOURCE` has
  their addresses and SHA-256s. The recipe, the patches and those assets together meet the LGPL's
  source obligation for it.

## Inside amdhip64_7.dll

Every row is at rocm-systems commit `7da43026e4f2ef7600433c38c820732f7a205d22`. On Windows the HIP
runtime (CLR) is built with ROCr's HSA runtime and AMD's PAL compiled in.

| Component | Version or commit | Inside | Licence | Files here (upstream name) | Source |
|---|---|---|---|---|---|
| amdgpu-windows-interop (the directory PAL comes from; its root licence) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll (see "AMD's EULA in amdhip64_7.dll") | AMD Software End User License Agreement (LicenseRef-AMD-EULA) | `amdgpu-windows-interop.EULA` (`LICENSE`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop |
| ELFIO (bundled in CLR, `rocclr/elf/elfio`) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `elfio.NOTICE` (the licence comment of `elfio.hpp`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/elf/elfio |
| HIP headers (HIP_COMMON_DIR) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `hip.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/hip |
| HIP runtime (CLR) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `hip-clr.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr |
| MetroHash (in PAL) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | Apache-2.0 | `metrohash.LICENSE` (`LICENSE`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/shared/metrohash |
| AMD PAL + DevDriver (prebuilt static libs, amdgpu-windows-interop/pal) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `pal.LICENSE` (`LICENSE`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal |
| ROCclr (part of CLR) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `rocclr.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr |
| ROCr hsa-runtime subtree | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | NCSA | `rocr-hsa-runtime.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime/runtime/hsa-runtime |
| ROCr / HSA runtime (compiled into CLR on Windows) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | NCSA | `rocr-runtime.LICENSE` (`LICENSE.txt`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime |

The first row is the root licence of `shared/amdgpu-windows-interop`, the directory PAL comes from.
It is AMD's End User License Agreement, not an open-source licence, and its text ships beside the
MIT and NCSA texts: see "AMD's EULA in amdhip64_7.dll".

## Inside rocm_kpack.dll

| Component | Version or commit | Inside | Licence | Files here (upstream name) | Source |
|---|---|---|---|---|---|
| msgpack-cxx 7.0.0 (header-only) | cpp-7.0.0 | rocm_kpack.dll | BSL-1.0 | `msgpack-cxx.LICENSE` (`LICENSE_1_0.txt`) | https://github.com/msgpack/msgpack-c |
| rocm-kpack | `7da43026e4f2ef7600433c38c820732f7a205d22` | rocm_kpack.dll | MIT | `rocm-kpack.LICENSE` (`LICENSE`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/kpack |
| zstd 1.5.7 (static, TheRock's bundled copy; dual BSD-3-Clause OR GPL-2.0-only, BSD option taken) | v1.5.7 | rocm_kpack.dll | BSD-3-Clause | `zstd.LICENSE` (`LICENSE`) | https://github.com/facebook/zstd |

## Inside the FFmpeg DLLs

Four components, each linked statically into the DLLs named. Which DLL holds each was read from
the DLLs' strings: zlib's `inflate 1.3.2` / `deflate 1.3.2`, dav1d's `libdav1d` and version,
libwebp's encoder, and the AMF interface.

| Component | Version or commit | Inside | Licence | Files here (upstream name) | Source |
|---|---|---|---|---|---|
| AMD AMF headers (headers only; the encoder is AMD's `amfrt64.dll` from the driver, loaded at run time, not shipped) | v1.5.2 (`AMF-headers-v1.5.2.tar.gz`) | avcodec-62.dll (`h264_amf`), avutil-60.dll (the AMF hwcontext) | MIT | `amf.LICENSE` (`LICENSE.txt`) | https://github.com/GPUOpen-LibrariesAndSDKs/AMF/releases/tag/v1.5.2 |
| dav1d | 1.5.3 | avcodec-62.dll | BSD-2-Clause | `dav1d.LICENSE` (`COPYING`); `dav1d.PATENTS` (`doc/PATENTS`) | https://code.videolan.org/videolan/dav1d (archive: https://github.com/videolan/dav1d/archive/1.5.3.tar.gz) |
| libwebp, with sharpyuv | 1.6.0 | avcodec-62.dll | BSD-3-Clause | `libwebp.LICENSE` (`COPYING`); `libwebp.PATENTS` (`PATENTS`) | https://chromium.googlesource.com/webm/libwebp (archive: https://github.com/webmproject/libwebp/archive/v1.6.0.tar.gz) |
| zlib | 1.3.2 | avcodec-62.dll, avformat-62.dll | Zlib | `zlib.LICENSE` (`LICENSE`) | https://github.com/madler/zlib (archive: https://github.com/madler/zlib/archive/v1.3.2.tar.gz) |

Notes on single rows:

- **Not inside: H.264 encoders.** `h264_amf` and `h264_mf` are FFmpeg's own LGPL wrapper code.
  They call encoders that are not in the package (see "H.264").
- **Patent texts.** `dav1d.PATENTS` is the Alliance for Open Media Patent License 1.0, which ends
  for a party that sues over the format. `libwebp.PATENTS` is Google's grant for WebP.
- **No GPL library is linked**, and no LGPL-3.0 one: FFmpeg is configured without
  `--enable-gpl`, `--enable-nonfree` and `--enable-version3`.

## Attribution these licences ask for in the documentation

**ROCr (NCSA).** `amdhip64_7.dll` contains AMD's ROCr HSA runtime, redistributed under the
University of Illinois/NCSA Open Source License. Its notice, as `rocr-runtime.LICENSE` has it
(`rocr-hsa-runtime.LICENSE` is the same text dated 2014-2020):

    Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.

    Developed by:

                    AMD Research and AMD HSA Software Development

                    Advanced Micro Devices, Inc.

                    www.amd.com

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal with the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

     - Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimers.
     - Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimers in
       the documentation and/or other materials provided with the distribution.
     - Neither the names of Advanced Micro Devices, Inc,
       nor the names of its contributors may be used to endorse or promote
       products derived from this Software without specific prior written
       permission.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS WITH THE SOFTWARE.

**dav1d (BSD-2-Clause) and libwebp (BSD-3-Clause).** A binary redistribution reproduces the
copyright notice, the conditions and the disclaimer "in the documentation and/or other materials
provided with the distribution": `dav1d.LICENSE` and `libwebp.LICENSE`, beside this file, do
that.

The other permissive licences here (MIT, Apache-2.0, BSL-1.0, BSD-3-Clause, zlib) ask that their
text accompany the binaries, which the files beside this one do.

## Where the source is

Brilliance links FFmpeg dynamically, through the five FFmpeg DLLs in `bin\`. Replacing all five
together with another build replaces FFmpeg, and every library inside them, in this program. The
replacement has to be ABI-compatible: the same DLL names (the same FFmpeg major versions), the
functions Brilliance imports, and the codecs it uses (the PNG, MJPEG, WebP, `h264_amf`, `h264_mf`
and AAC encoders, the MP4 muxer, and the decoders for the inputs you send).

**FFmpeg and the libraries inside its DLLs.**

- **FFmpeg's corresponding source** is the `n8.1.3` tag archive with the patches of vcpkg's
  `ffmpeg` port, applied by `win\ffmpeg\ports\ffmpeg\portfile.cmake` in the order it lists. The
  FFmpeg zip carries the whole recipe under `source\`, so the archive plus `source\ports\ffmpeg`
  is the complete source, and `source\build-ffmpeg.ps1` the scripts that build it.
  - **Archive:** https://github.com/ffmpeg/ffmpeg/archive/n8.1.3.tar.gz,
    SHA-256 `09f990289327d3ebfedf4fc15cae3f884a65dab7a5aac7bbb5b01ff83900efd9`,
    SHA-512 `8c667090bcdc70796caaa8cdc2cf82e57d7c7f6feedcb0ec7251cefd6a0c63b7cad203a86c800d3e2f69ca91255167aa4e631701060b96d576d1a9666ca7aafe`.
    The tag is also at https://git.ffmpeg.org/ffmpeg.git.
  - **FFmpeg is modified** by those patches. Each is vcpkg's, unedited:

    | Patch | Touches | What it does |
    |---|---|---|
    | 0002, 0004, 0007, 0024, 0046, 0047, 0049, 0050 | `configure` | MSVC linking, detection and UTF-8; library naming; dependency and pkg-config handling |
    | 0003 | `fftools/cmdutils.c` | a Windows include, for the programs |
    | 0005 | `libavcodec/x86/*.asm` | nasm compatibility |
    | 0013 | `libavcodec/mf_utils.c` | defines `WINVER` for the Media Foundation code |
    | 0040 | `libavformat/avformat.h`, `mux_utils.c` | adds `av_stream_get_first_dts`, an API for Chromium; the engine does not call it |
    | 0045 | `ffbuild/common.mak` | uses a prebuilt `bin2c` host tool |
    | 0048 | `configure`, `libavformat`, `libavcodec` cbs files | FFmpeg's upstream change 23039, backported |

  - **The configure switches** are vcpkg's per feature plus the "Brilliance" block in the same
    `portfile.cmake`. The full line is in the zip's `BUILDINFO.txt` and in each DLL
    (`avcodec_configuration()`). `ffmpeg.SOURCE` names the zip and its SHA-256.
- **The libraries' source** is the archive each row above names, at the version the vcpkg
  registry tag pins: each port's `portfile.cmake` in `microsoft/vcpkg` at commit
  `9e593bb18ea69cc5095e012465dcd675a822ed0d` holds its SHA-512. Their code is built unmodified.
  The ports' patches touch only build files:
  - zlib's `0001-no-version-script-on-ohos.patch`: `CMakeLists.txt`, for OpenHarmony;
  - libwebp's four: `CMakeLists.txt`, `cmake/cpu.cmake`, `cmake/WebPConfig.cmake.in`. They
    include CMake packaging, SDL and Android options, and MSVC SIMD flags.

  dav1d and the AMF headers have no patches. To use a modified version of one, change its port
  or source in the recipe, rebuild, and replace the DLLs in `bin\`.

**The HIP runtime and rocm_kpack.** `amdhip64_7.dll` and `rocm_kpack.dll` are AMD's binaries from
the `rocm_sdk_core` 10.1.0 wheel, unmodified. Their source and build recipe:

- TheRock, commit `b324869fb3116353ac7eec5e6926aa3d452ac987`:
  https://github.com/ROCm/TheRock/tree/b324869fb3116353ac7eec5e6926aa3d452ac987
- rocm-systems, commit `7da43026e4f2ef7600433c38c820732f7a205d22`:
  https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22
  (`projects/clr`, `projects/hip`, `projects/rocr-runtime`, `shared/kpack`,
  `shared/amdgpu-windows-interop`).
- msgpack-cxx `cpp-7.0.0` (https://github.com/msgpack/msgpack-c) and zstd `v1.5.7`
  (https://github.com/facebook/zstd), which TheRock builds into `rocm_kpack.dll`.

## Not covered yet

- **PAL's prebuilt static libraries** (`zstd.lib`, `pal_lz4.lib`, `mpack.lib`, `cwpack.lib`,
  `dd_libyaml.lib`, `stb_sprintf.lib`, `amdrdf.lib`, `pal_uuid.lib`). `amdhip64_7.dll` shows no
  strings from them, so they were probably not linked in; their upstream versions are unknown and
  no texts were collected.
- **Apache-2.0 NOTICE files.** MetroHash's tree was not checked for one.
- **If more of FFmpeg is added.** The package ships no `avfilter`, `avdevice` or
  `ffmpeg`/`ffprobe`/`ffplay`, and the FFmpeg build links only the four libraries above. A
  library or encoder added to the recipe (`win\ffmpeg`) brings its licence here first.

## AMD's EULA in amdhip64_7.dll

- **The package keeps `amdhip64_7.dll`** and ships the EULA text beside the MIT and NCSA texts.
  The package is for AMD GPUs.
- **Where the EULA comes from.** On Windows, TheRock builds the HIP runtime (CLR) with
  `ROCCLR_ENABLE_PAL=1` and `AMD_COMPUTE_WIN` set to rocm-systems `shared/amdgpu-windows-interop`.
  That directory's root `LICENSE`, shipped unedited as `amdgpu-windows-interop.EULA`, is the "AMD
  Software End User License Agreement", not an open-source licence. Its section 2 grants use in
  object code only, with systems that include AMD products, and lets Derivative Works be
  distributed only in object code, for such systems, under terms at least as restrictive as its
  own.
- **The `pal/` subtree** of the same directory has its own MIT `LICENSE` (`pal.LICENSE` here), and
  MetroHash inside it is Apache-2.0 (`metrohash.LICENSE`).
- **Which licence covers PAL's libraries is not stated.** `amdhip64_7.dll` contains PAL and
  DevDriver code. PAL's libraries are prebuilt, fetched by the build, and nothing found states
  which of the two licences covers them.
- **AMD publishes this DLL itself** in the public `rocm_sdk_core` wheel, which carries no licence
  file for it.

## Where each text was fetched

18 files. The SHA-256 is of the upstream file as fetched; every file here except the extract
(`elfio.NOTICE`) has that hash. `llvm-openmp.LICENSE` and `msvc-runtime.NOTICE` come from Visual
Studio Build Tools, not from a fetch.

| File | Fetched from | SHA-256 of the upstream file |
|---|---|---|
| `amdgpu-windows-interop.EULA` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/LICENSE | `154bdf053ef72a3cbf01d0065cfcbc0188745de81b914f744f90a0bb8015d18a` |
| `amf.LICENSE` | https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AMF/v1.5.2/LICENSE.txt (downloaded by vcpkg's `amd-amf` port) | `eb297397aaa455b5668ab67d216b83828466152dab123fa92384c6ec16b74170` |
| `dav1d.LICENSE` | `dav1d-1.5.3/COPYING` in `videolan-dav1d-1.5.3.tar.gz` (https://github.com/videolan/dav1d/archive/1.5.3.tar.gz, SHA-256 `cbe212b02faf8c6eed5b6d55ef8a6e363aaab83f15112e960701a9c3df813686`) | `dd92c3c2247c5651606fc23a5e2d6a1ebc5ace9a3e49cbde0e12f05ad1cb1ee5` |
| `dav1d.PATENTS` | `dav1d-1.5.3/doc/PATENTS` in the same archive | `335eca574598bf4ca181b12f708d6669e5a5e78c8e1513e5b35fa1f03901484b` |
| `elfio.NOTICE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/elf/elfio/elfio.hpp (whole header; the file here is its leading comment) | `e8f13ac81da4e93b9d44f4f9c1eb5ef6982396f59cf61d057b12386c91802eca` |
| `hip-clr.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/LICENSE.md | `b185aaa652b0bf066c37a0d6314ce4bf4521e4a3c9bf46edd2f6a777ac522223` |
| `hip.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/hip/LICENSE.md | `b185aaa652b0bf066c37a0d6314ce4bf4521e4a3c9bf46edd2f6a777ac522223` |
| `libwebp.LICENSE` | `libwebp-1.6.0/COPYING` in `webmproject-libwebp-v1.6.0.tar.gz` (https://github.com/webmproject/libwebp/archive/v1.6.0.tar.gz, SHA-256 `93a852c2b3efafee3723efd4636de855b46f9fe1efddd607e1f42f60fc8f2136`) | `5aec868f669e384a22372a4e8a1a6cd7d44c64cd451f960ca69cc170d1e13acf` |
| `libwebp.PATENTS` | `libwebp-1.6.0/PATENTS` in the same archive | `cc3273e0694ea5896145e0677699b53471b03ea43021ddc50e7923fbb9f5023c` |
| `metrohash.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/shared/metrohash/LICENSE | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| `msgpack-cxx.LICENSE` | https://raw.githubusercontent.com/msgpack/msgpack-c/cpp-7.0.0/LICENSE_1_0.txt | `c9bff75738922193e67fa726fa225535870d2aa1059f91452c411736284ad566` |
| `pal.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/LICENSE | `0774a30583c3e28700537f2f3ca225128c271935baea7cbbed84a1a94a3c3f8b` |
| `rocclr.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/LICENSE.md | `b185aaa652b0bf066c37a0d6314ce4bf4521e4a3c9bf46edd2f6a777ac522223` |
| `rocm-kpack.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/shared/kpack/LICENSE | `e071e8c7d920817b6415f4b87c115c6aefafe137e36843b3723f28dde5f484c3` |
| `rocr-hsa-runtime.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime/runtime/hsa-runtime/LICENSE.md | `dac0a5360f50972cceaed16767bc9e94642aa497d558a74471e842f7192c7403` |
| `rocr-runtime.LICENSE` | https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime/LICENSE.txt | `ffa5a77ce21419e276bd9068faec94333128e49e1c95426d9c1d35435e8fe835` |
| `zlib.LICENSE` | `zlib-1.3.2/LICENSE` in `madler-zlib-v1.3.2.tar.gz` (https://github.com/madler/zlib/archive/v1.3.2.tar.gz, SHA-256 `b99a0b86c0ba9360ec7e78c4f1e43b1cbdf1e6936c8fa0f6835c0cd694a495a1`) | `e32ff4e00d9d94930537635291da39e7e612703334bf6fde8c7f1686fe8a45a2` |
| `zstd.LICENSE` | https://raw.githubusercontent.com/facebook/zstd/v1.5.7/LICENSE | `7055266497633c9025b777c78eb7235af13922117480ed5c674677adc381c9d8` |
