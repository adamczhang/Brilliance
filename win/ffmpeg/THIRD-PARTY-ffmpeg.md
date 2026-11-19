# Third-party software in Brilliance's FFmpeg build

This is the licence list for `brilliance-ffmpeg-8.1.3-win64-lgpl-<date>.zip`, the FFmpeg that
`win/ffmpeg` builds (README.md there). It follows `win/licenses/THIRD-PARTY.md`'s format.

- **What happened to it.** The integration (2026-11-19) replaced the FFmpeg sections of
  `THIRD-PARTY.md` ("The DLLs" row, "Inside the FFmpeg DLLs" and "Where the source is") and
  `win/licenses/ffmpeg.SOURCE` with this, and removed the texts only BtbN's build needed.
- **Where the files are.** In the zip: `LICENSE.txt` (FFmpeg's) and `licenses\`. Four of the
  texts are already in `win/licenses`, byte for byte (the last section). Two are new:
  `ffmpeg.LICENSE`, the LGPL-2.1 text, and `dav1d.PATENTS`.
- **The texts are upstream's, unedited.** The build script takes them from the same source
  archives vcpkg builds from, and the AMF headers' licence from the file vcpkg's `amd-amf` port
  downloads. A text that refers to `COPYING` means the file the table lists under that upstream
  name.
- **Collected 2026-11-18**, from the build of that day.

## The DLLs

| DLLs in `bin\` | What they are | Built from | Licence | Files |
|---|---|---|---|---|
| `avcodec-62.dll`, `avformat-62.dll`, `avutil-60.dll`, `swresample-6.dll`, `swscale-9.dll` | FFmpeg 8.1.3 | `n8.1.3` (https://github.com/ffmpeg/ffmpeg/archive/n8.1.3.tar.gz) with vcpkg's `ffmpeg` port patches, built by `win/ffmpeg` (vcpkg registry tag 2026.07.29, MSVC 14.51.36231) | LGPL-2.1-or-later (configured without `--enable-gpl`, `--enable-nonfree` or `--enable-version3`; each DLL reports "LGPL version 2.1 or later") | `ffmpeg.LICENSE` (also `LICENSE.txt`), the libraries below, and "Where the source is" |

The DLLs link no GCC, MinGW or winpthreads runtime. They import Windows system DLLs, the
Universal CRT and the VC++ runtime (`vcruntime140.dll`), which `THIRD-PARTY.md` already lists
(`msvc-runtime.NOTICE`).

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

**Not inside: H.264 encoders.** The two H.264 encoders the build registers, `h264_amf` and
`h264_mf`, are FFmpeg's own LGPL wrapper code. They call encoders that are not in the package:
- AMD's hardware encoder, through the AMF runtime the AMD driver installs;
- the H.264 encoder Windows Media Foundation provides.

Those makers license the codec. No H.264 encoder implementation ships: no OpenH264 and no x264.
The new `avcodec-62.dll` contains none of OpenH264's `Wels*` symbols, where BtbN's has 66.

**Decoding is unchanged.** H.264 and HEVC input is still decoded by FFmpeg's built-in decoders,
and no patent licence is supplied for that (`THIRD-PARTY.md`, "H.264").

## Attribution these licences ask for in the documentation

- **dav1d (BSD-2-Clause) and libwebp (BSD-3-Clause):** a binary redistribution reproduces the
  copyright notice, the conditions and the disclaimer "in the documentation and/or other
  materials provided with the distribution". `dav1d.LICENSE` and `libwebp.LICENSE`, shipped
  beside `THIRD-PARTY.md`, do that.
- **AMF headers (MIT):** the notice and permission text ship in `amf.LICENSE`.
- **zlib:** the licence asks for nothing in a binary's documentation; the text ships anyway.
- **FFmpeg (LGPL-2.1-or-later):** the licence text (`ffmpeg.LICENSE.txt` in the package, from
  the zip's `LICENSE.txt`), a notice that the program uses FFmpeg under it, and where its source
  is. That is `ffmpeg.SOURCE`, which the integration task rewrites from the next section.

## Where the source is

**FFmpeg's corresponding source** is the `n8.1.3` tag archive with the patches of vcpkg's
`ffmpeg` port, applied by `win/ffmpeg/ports/ffmpeg/portfile.cmake` in the order it lists. The zip
carries the whole recipe under `source\`, so the archive plus `source\ports\ffmpeg` is the
complete source, and `source\build-ffmpeg.ps1` the scripts that build it.
- **Archive:** https://github.com/ffmpeg/ffmpeg/archive/n8.1.3.tar.gz,
  SHA-256 `09f990289327d3ebfedf4fc15cae3f884a65dab7a5aac7bbb5b01ff83900efd9`,
  SHA-512 `8c667090bcdc70796caaa8cdc2cf82e57d7c7f6feedcb0ec7251cefd6a0c63b7cad203a86c800d3e2f69ca91255167aa4e631701060b96d576d1a9666ca7aafe`.
  The tag is also at https://git.ffmpeg.org/ffmpeg.git.
- **FFmpeg is modified** by those patches, unlike BtbN's build, so `ffmpeg.SOURCE`'s line
  "Brilliance does not modify FFmpeg" has to change. Each patch is vcpkg's, unedited:

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
  (`avcodec_configuration()`).

**The libraries' source** is the archive each row names, at the version the registry tag pins:
each port's `portfile.cmake` in `microsoft/vcpkg` at commit
`9e593bb18ea69cc5095e012465dcd675a822ed0d` holds its SHA-512. Their code is built unmodified.
The ports' patches touch only build files:
- zlib's `0001-no-version-script-on-ohos.patch`: `CMakeLists.txt`, for OpenHarmony;
- libwebp's four: `CMakeLists.txt`, `cmake/cpu.cmake`, `cmake/WebPConfig.cmake.in`. They
  include CMake packaging, SDL and Android options, and MSVC SIMD flags.

dav1d and the AMF headers have no patches.

The program links these DLLs dynamically: replacing them with a build of the same FFmpeg major
versions replaces FFmpeg in it.

## Where each text was taken

| File | Taken from | SHA-256 | Already in `win/licenses`? |
|---|---|---|---|
| `ffmpeg.LICENSE` (also the zip's `LICENSE.txt`) | `FFmpeg-n8.1.3/COPYING.LGPLv2.1` in `ffmpeg-ffmpeg-n8.1.3.tar.gz` | `246041b6ecf9bc32d718a62c57877c78b5eb397b6467e74ed7ae2626ab189c30` | no (BtbN's `ffmpeg.LICENSE.txt` was the LGPL-3.0 text) |
| `amf.LICENSE` | https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AMF/v1.5.2/LICENSE.txt (downloaded by the `amd-amf` port) | `eb297397aaa455b5668ab67d216b83828466152dab123fa92384c6ec16b74170` | yes, identical |
| `dav1d.LICENSE` | `dav1d-1.5.3/COPYING` in `videolan-dav1d-1.5.3.tar.gz` | `dd92c3c2247c5651606fc23a5e2d6a1ebc5ace9a3e49cbde0e12f05ad1cb1ee5` | yes, identical |
| `dav1d.PATENTS` | `dav1d-1.5.3/doc/PATENTS` in the same archive | `335eca574598bf4ca181b12f708d6669e5a5e78c8e1513e5b35fa1f03901484b` | no |
| `libwebp.LICENSE` | `libwebp-1.6.0/COPYING` in `webmproject-libwebp-v1.6.0.tar.gz` | `5aec868f669e384a22372a4e8a1a6cd7d44c64cd451f960ca69cc170d1e13acf` | yes, identical |
| `libwebp.PATENTS` | `libwebp-1.6.0/PATENTS` in the same archive | `cc3273e0694ea5896145e0677699b53471b03ea43021ddc50e7923fbb9f5023c` | yes, identical |
| `zlib.LICENSE` | `zlib-1.3.2/LICENSE` in `madler-zlib-v1.3.2.tar.gz` | `e32ff4e00d9d94930537635291da39e7e612703334bf6fde8c7f1686fe8a45a2` | yes, identical |

The archives, as vcpkg downloaded them on 2026-11-18:

| Archive | SHA-256 |
|---|---|
| `ffmpeg-ffmpeg-n8.1.3.tar.gz` | `09f990289327d3ebfedf4fc15cae3f884a65dab7a5aac7bbb5b01ff83900efd9` |
| `videolan-dav1d-1.5.3.tar.gz` | `cbe212b02faf8c6eed5b6d55ef8a6e363aaab83f15112e960701a9c3df813686` |
| `webmproject-libwebp-v1.6.0.tar.gz` | `93a852c2b3efafee3723efd4636de855b46f9fe1efddd607e1f42f60fc8f2136` |
| `madler-zlib-v1.3.2.tar.gz` | `b99a0b86c0ba9360ec7e78c4f1e43b1cbdf1e6936c8fa0f6835c0cd694a495a1` |
| `AMF-headers-v1.5.2.tar.gz` | `d3c12eb324edf05e214608b6a395a51dd95770ed9d45520185d6c3a206811c99` |

## What leaves `THIRD-PARTY.md` with BtbN's build

For the integration task:
- **Every row of "Inside the FFmpeg DLLs" goes,** except dav1d, libwebp, zlib and the AMF
  headers: 72 of its 76 components, including OpenH264, the GCC runtime, winpthreads and the
  MinGW-w64 runtime.
- **The "Rust crates inside avcodec-62.dll" section goes.** rav1e is not built.
- **The texts used only by those rows go:**
  - `GPL-3.0.txt` stays only if something else still needs it. FFmpeg is now LGPL-2.1, and
    GMP and aribb24 are gone (BtbN's build needed it for its LGPL-3.0 libraries).
  - `ffmpeg.LICENSE.txt` changes from the LGPL-3.0 text to the LGPL-2.1 text.
  - `openh264.LICENSE` and its rows go (`THIRD-PARTY.md`, "H.264").
- **`dav1d.PATENTS` is added.**
- **The amf row's version changes** from AMF commit `57ee0cfb...` to the v1.5.2 headers. The
  licence text is the same.
