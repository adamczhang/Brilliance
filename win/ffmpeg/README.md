# FFmpeg for Brilliance on Windows

This directory is the recipe for the FFmpeg the Windows package ships. It builds FFmpeg 8.1.3 with
the repository's pinned compiler (Visual Studio Build Tools 18, MSVC 14.51) through vcpkg. The
result is the five DLLs the engine links, `avcodec`, `avformat`, `avutil`, `swscale` and
`swresample`, published as a release asset:

    brilliance-ffmpeg-8.1.3-win64-lgpl-<yyyyMMdd>.zip     (and .zip.sha256)

It replaces BtbN's LGPL build (`ffmpeg-n8.1.3-35-gf1acb0a266-win64-lgpl-shared-8.1.zip`), which
carries OpenH264 and seven hardware H.264 encoders. It follows three decisions:
- the FFmpeg is a release asset plus this recipe;
- H.264 output goes only through AMD AMF or Windows Media Foundation;
- video generation returns through them.

## What is in it

**LGPL-2.1-or-later.** Configured without `--enable-gpl`, `--enable-nonfree` or
`--enable-version3`. Each DLL reports "LGPL version 2.1 or later".

**Shared FFmpeg, static dependencies.** The triplet `x64-windows-brl` builds every dependency as
a static library and FFmpeg alone as DLLs. The DLLs import:
- Windows system DLLs: `KERNEL32`, `ole32` and `bcrypt`;
- the Universal CRT and the VC++ runtime (`/MD`), which the package already ships.

Three DLLs are loaded at run time, only when an H.264 encoder opens, and none is linked:
- Media Foundation's `mfplat.dll` and `d3d11.dll`, which come with Windows;
- AMD's AMF runtime, `amfrt64.dll`, from the AMD driver.

| Library | Version | In | Why |
|---|---|---|---|
| FFmpeg | 8.1.3 (`n8.1.3` tag archive, with vcpkg's port patches) | all five DLLs | the media path (`core/mm`) |
| dav1d | 1.5.3 | `avcodec-62.dll` | AV1 decoding (FFmpeg's own AV1 decoder needs a hardware device) |
| libwebp (with sharpyuv) | 1.6.0 | `avcodec-62.dll` | WebP encoding for `/v1/images` and video thumbnails (WebP *decoding* is FFmpeg's own) |
| zlib | 1.3.2 | `avcodec-62.dll`, `avformat-62.dll` | PNG, and compressed MP4/Matroska headers |
| AMD AMF headers | 1.5.2 | `avcodec-62.dll`, `avutil-60.dll` | the interface `h264_amf` calls; headers only, no AMD code |

### Encoders: exactly six

| Encoder | For | What it is |
|---|---|---|
| `png`, `mjpeg` | `/v1/images` PNG and JPEG | FFmpeg's own |
| `libwebp` | `/v1/images` WebP, video thumbnails | libwebp |
| `h264_amf` | `/v1/videos`, first choice | a thin wrapper that calls AMD's hardware H.264 encoder through the AMF runtime in the AMD driver |
| `h264_mf` | `/v1/videos`, second choice | a thin wrapper that calls the H.264 encoder Windows Media Foundation provides |
| `aac` | the `/v1/videos` soundtrack | FFmpeg's own AAC encoder (`core/mm/mp4.cpp` muxes one when the clip has audio) |

**The package ships no H.264 encoder implementation of its own.** It can only call AMD's or
Windows' encoder, whose makers license the codec. There is no libopenh264, no x264, and no other
hardware H.264 encoder (NVENC, QSV, D3D12VA, VAAPI, Vulkan). There is no other video encoder
(no HEVC, AV1, VP9 or MPEG-4) and no audio encoder but `aac`. Music generation and the speech
APIs write WAV/PCM themselves.

### What video generation needs, and has

What video generation needs (`core/mm/mp4.cpp` `Mp4Writer`, `core/server/videos.cpp`):
- an H.264 encoder: `h264_amf`, then `h264_mf`;
- `aac` for the soundtrack;
- the `mp4` muxer (FFmpeg's `mov` muxer, which it is built on), writing to memory through the
  writer's own AVIO, with `+faststart`'s second pass reading back through its `io_open`;
- swscale's RGB24 to YUV420P (limited-range BT.709);
- `libwebp` for the thumbnail.

LTX-2's `input_reference` round trip (`core/mm/media.cpp`, `h264_roundtrip`) needs an H.264
encoder and the `h264` decoder.

Since 2026-11-19 the engine asks for `h264_amf`, then `h264_mf`, by name, in both places and
takes no other H.264 encoder (`core/mm/mp4.h`). `BRILLIANCE_H264_ENCODER` narrows the two (the
host tests set `h264_mf`, Windows' software encoder, so they stay off the card).

### Decoders, demuxers, parsers and bitstream filters

These are FFmpeg's full default set, without the four AMF decoders, so decoding stays on
FFmpeg's own decoders and dav1d. The engine chooses which demuxers a request may reach (`kFormats`
and `kAudioFormats` in `core/mm/media.cpp`). Among the decoders:
- video: H.264 (FFmpeg's built-in decoder), HEVC, VP8, VP9, AV1 (dav1d), MPEG-4, Theora, ProRes;
- pictures: PNG, APNG, JPEG, JPEG-LS, WebP, GIF, BMP, TIFF, QOI, PNM, AVIF (through `mov` and
  dav1d);
- audio: PCM, ADPCM, MP3, FLAC, Vorbis, Opus, AAC, ALAC.

### Muxers: `mp4` and `mov` only

### Not built

- protocols, the network, devices, avdevice and avfilter: the engine reads and writes memory
  through its own AVIO;
- hardware decoding (`--disable-hwaccels`), and every other hardware path: d3d12va, dxva2, CUDA,
  NVENC/NVDEC, QSV/VPL, VAAPI, VDPAU, Vulkan and OpenCL;
- d3d11va is on only as the hwcontext FFmpeg 8.1.3's Media Foundation encoder needs to compile
  (`libavcodec/mfenc.c` uses its types unconditionally). It adds no encoder and no decoder;
- JPEG XL: FFmpeg 8.1 decodes it only through libjxl, which this build leaves out. A JPEG XL
  picture is refused with an error. BtbN's build read it.

The exact configure line is in the zip's `BUILDINFO.txt`.

## The files

| File | What it is |
|---|---|
| `build-ffmpeg.ps1` | Fetches the pinned registry, builds, stages and zips. |
| `vcpkg.json` | The manifest: FFmpeg's features (`avcodec avformat swscale swresample dav1d webp zlib amf ffprobe`), and the registry baseline. |
| `ports/ffmpeg/` | The overlay port: vcpkg's `ffmpeg` port 8.1.2#3 from the registry tag, moved to 8.1.3 (`REF`/`SHA512` in `portfile.cmake`, `version` in `vcpkg.json`), with the "Brilliance" configure block before `Building Options`. |
| `triplets/x64-windows-brl.cmake` | x64, dynamic CRT, static libraries, FFmpeg shared, release only, MSVC 14.51. |
| `check-ffmpeg.c` | A standalone check of the built DLLs (below). |
| `THIRD-PARTY-ffmpeg.md` | The licences of the zip's contents. |

## Pins

| What | Pin |
|---|---|
| vcpkg registry | `microsoft/vcpkg` tag `2026.07.29`, commit `9e593bb18ea69cc5095e012465dcd675a822ed0d` (the last release tag whose `ffmpeg` port is 8.1.x) |
| vcpkg tool | 2026-07-27 (`98d7cb0c`), the one bundled with Visual Studio Build Tools 18 and the one the tag names; copied into the registry clone, not downloaded |
| FFmpeg source | `https://github.com/ffmpeg/ffmpeg/archive/n8.1.3.tar.gz`, SHA-512 `8c667090...9666ca7aafe` (in `portfile.cmake`) |
| dav1d, libwebp, zlib, AMF headers, and the build tools | as the registry tag pins them (each port's `portfile.cmake` holds its SHA-512) |
| Compiler | MSVC 14.51.36231, Windows SDK 10.0.26100.0 (`.toolchain\toolchain.json`) |

## Building it

Needs: Visual Studio Build Tools 18 with MSVC 14.51.36231 (the tested toolset `win\setup.ps1`
prefers; this recipe takes no other, though `win\setup.ps1` accepts a newer 14.x for the engine
build), git, and the network for the first run. No GPU, no administrator.

    powershell -NoProfile -ExecutionPolicy Bypass -File win\ffmpeg\build-ffmpeg.ps1 -Clean

- **Where things go.** Everything stays inside the checkout, all of it git-ignored:
  - the registry: `.toolchain\vcpkg`;
  - downloads: `.toolchain\vcpkg-downloads`;
  - vcpkg's trees: `.toolchain\ffmpeg-vcpkg\{installed,bt,pk}`;
  - the binary cache: `.toolchain\vcpkg-archives`;
  - the package and zip: `artifacts\ffmpeg`.
- **The B: drive.** FFmpeg's port refuses a source path with a space, and the checkout is
  `Brilliance 2`. The script maps the checkout to `B:` with `subst` for the length of the vcpkg
  run, and removes the mapping afterwards. Use `-Drive X` if B: is taken.
- **What the first run downloads,** into `.toolchain\vcpkg-downloads` (measured 2026-11-18):
  - sources, about 41 MB: FFmpeg 8.1.3, FFmpeg 8.1.1 (for the port's `bin2c` host tool), dav1d,
    libwebp, zlib, pkgconf and the AMF headers;
  - the msys2 packages FFmpeg's configure runs under, about 27 MB;
  - nasm, meson and Python, about 17 MB;
  - CMake 4.4.0, 7-Zip and PowerShell 7.6.3, about 175 MB, which this vcpkg tool fetches for
    itself.

  That is about 260 MB in all, 888 MB once unpacked. The registry is a 9 MB depth-1 fetch, 68 MB
  checked out.
- **How long.** On the test machine, 3.3 minutes for `-Clean` with the downloads already present
  (2026-11-18, 7:13-7:16 AM), and about 5 minutes for the first run, downloads included. Without
  `-Clean`, a rerun restores the packages from the binary cache in seconds and only re-zips.

**The zip.** One top folder, `brilliance-ffmpeg-8.1.3-win64-lgpl-<date>\`, holding:
- `bin\` the five DLLs;
- `lib\` their import libraries;
- `include\` the headers;
- `LICENSE.txt`, FFmpeg's LGPL-2.1 text;
- `licenses\` every component's licence;
- `source\` this recipe, with the port's patches;
- `BUILDINFO.txt`: the pins, the configure line, `ffprobe -buildconf` and `vcpkg list`.

That is the layout `win\setup.ps1` unpacks BtbN's archive into (`bin`, `lib\<name>.lib`,
`include\libav*`, `LICENSE.txt`), so `brl_find_ffmpeg` (`win/cmake/RadWindows.cmake`) reads it
unchanged.

**The zip is not byte-reproducible.** Its entries are sorted and carry the date in the name as
their timestamp, but the DLLs are not: MSVC writes a timestamp and the build path into each
link. Two builds of the same recipe match in configuration and behaviour, not in bytes. The
release asset is pinned by its own SHA-256.

## Checking it

`check-ffmpeg.c` uses the DLLs as the engine does. It fails on any of these:
- a licence other than LGPL-2.1-or-later;
- `--enable-gpl`, `--enable-nonfree`, libopenh264 or libx264 in the configure line;
- any encoder outside png, mjpeg, libwebp, h264_amf, h264_mf and aac. That covers libopenh264,
  libx264 and every other H.264 or video encoder; a list of them is also looked up by name;
- anything but exactly h264_amf and h264_mf for H.264;
- any muxer but mp4/mov;
- any protocol;
- an AMF decoder;
- a missing decoder from the engine's list;
- AV1 not resolving to dav1d.

It then runs three tests:
- **Decode.** It decodes every file in a directory from memory, through a custom AVIO, an empty
  protocol whitelist and the engine's demuxer whitelist, to RGB24 or 16 kHz mono float.
- **Pictures.** It encodes a picture as PNG, JPEG and WebP and decodes each back. The PNG must
  be pixel-exact.
- **Video.** It writes an MP4 the way `Mp4Writer` does:
  - RGB24 through swscale to BT.709 YUV420P;
  - `h264_mf` on Windows' software encoder (`hw_encoding` 0, so no GPU);
  - an `aac` soundtrack;
  - the `mp4` muxer in memory with `+faststart`.

  It then decodes the file back.

`h264_amf` is not opened: it runs on an AMD GPU. Integration tests it on a card. The check
reports whether the driver's `amfrt64.dll` is installed.

From a VS x64 developer prompt (`vcvars64.bat -vcvars_ver=14.51`), with `PKG` the unpacked folder:

    cl /nologo /O2 /std:c17 /I %PKG%\include check-ffmpeg.c /link /LIBPATH:%PKG%\lib avcodec.lib avformat.lib avutil.lib swscale.lib swresample.lib
    set PATH=%PKG%\bin;%PATH%
    check-ffmpeg.exe <media-dir> <out-dir> [--no-h264-mf]

`tests\data\media` has the pictures and clips. For the first check, audio files (wav, ADPCM wav,
mp3, flac, Vorbis, Opus in Ogg and WebM, AAC, m4a, aiff) were generated with BtbN's `ffmpeg.exe`.

## Using it

Integrated 2026-11-19:
- `win\setup.ps1` pins the zip by its SHA-256 and unpacks it into `.toolchain\ffmpeg`, replacing
  an earlier install (a stamp, `.toolchain\ffmpeg\.archive-sha256`, names the archive it came
  from). It downloads the zip from the v0.3.0-alpha.1 release, or copies it from
  `artifacts\ffmpeg`, where `build-ffmpeg.ps1` leaves it, when a copy is there. A rebuilt zip
  has a new SHA-256: publish it with its `.sha256` and change the pin's `url` and `sha` together.
- `win\licenses\ffmpeg.SOURCE` and the FFmpeg sections of `win\licenses\THIRD-PARTY.md` come from
  `THIRD-PARTY-ffmpeg.md`; the texts only BtbN's build needed are gone.
- `core/mm/mp4.cpp` and the LTX-2 round trip in `core/mm/media.cpp` take `h264_amf`, then
  `h264_mf`, never another H.264 encoder.
- Still to do: video generation qualified with a supported model, and `h264_amf` opened on a
  card.
