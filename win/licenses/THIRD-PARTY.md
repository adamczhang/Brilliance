# Third-party software in the Brilliance Windows package

This index records the third-party DLLs in the package's `bin\`, the evidence for what they
contain and the files that carry their licence terms and notices.

- **Where the files are.** In the package: `share\doc\radiance\third_party`, beside this file. In
  the source tree: `win\licenses`. The build adds two more texts from what it was built against:
  `ffmpeg.LICENSE.txt` (FFmpeg's LGPL-2.1 text, the FFmpeg zip's `LICENSE.txt`) and
  `amd_comgr.LICENSE.txt` (from the ROCm SDK, for packages with the HIP runtime). CPU-only
  packages carry the same source-tree notices and omit the HIP DLLs.
- **What is not here.** The engine's own third-party code (cpp-httplib, yyjson, llama.cpp and the
  rest) is named in `NOTICE`, one directory up, and its texts are beside this file too.
- Standalone texts retain upstream's bytes. The notice files and the source table identify
  extracts and compilations of upstream licence comments. A text that refers to `COPYING` or
  `LICENSES/...` means the file the table lists under that upstream name.
- **Collected 2026-11-17**; the FFmpeg rows 2026-11-18, from the build of Brilliance's own
  FFmpeg (`win\ffmpeg\THIRD-PARTY-ffmpeg.md`). The last section records where each text was
  fetched and its SHA-256.

## The DLLs

| DLLs in `bin\` | What they are | Built from | Licence | Files |
|---|---|---|---|---|
| `amdhip64_7.dll` | AMD's HIP runtime (HIP 7.16.26385) | AMD's `rocm_sdk_core` 10.1.0 wheel, TheRock `b324869fb3116353ac7eec5e6926aa3d452ac987` and rocm-systems `7da43026e4f2ef7600433c38c820732f7a205d22` | Component terms and Windows interop EULA below. The complete DLL's redistribution basis remains unresolved. | listed below |
| `rocm_kpack.dll` | ROCm's kernel-pack reader | the same wheel and commits (rocm-systems `shared/kpack`) | MIT, with BSL-1.0 and BSD-3-Clause parts | listed below |
| `amd_comgr.dll` | AMD's code-object manager | the same wheel, ROCm LLVM `d6f6cb691863e9258034ab48c1dfce21689d5606` | Apache-2.0 WITH LLVM-exception, legacy NCSA and linked component terms | `amd_comgr.LICENSE.txt` and the files below |
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

- **No H.264 encoder implementation ships in Brilliance 0.3.1.** OpenH264 and x264 are not in
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
  in 0.3.1, and music generation, which shares its route, is not served; music writes WAV or raw
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
- **The source of the package's FFmpeg.** It is built with vcpkg from the recipe in the source tree's
  `win\ffmpeg\`, as a zip pinned by its SHA-256 (`win\setup.ps1`). The zip and FFmpeg's own
  `n8.1.3` source archive are assets of the 0.3.0-alpha.1 release
  (https://github.com/adamczhang/Brilliance/releases/tag/v0.3.0-alpha.1); `ffmpeg.SOURCE` has
  their addresses and SHA-256s. The recipe, the patches and those assets together meet the LGPL's
  source obligation for it.

## AMD ROCm Core SDK 10.1.0 artifacts

AMD published all three files in the Windows x64 wheel
`rocm_sdk_core-10.1.0-py3-none-win_amd64.whl` (SHA-256
`b12f1c4cde14ed1cd8864006d17586c6b1e6a4e4695b9597cfe04ed7715e952f`). Brilliance stages them
byte-for-byte from the SDK. The hashes below match the wheel entries and the staged
0.3.0-alpha.5 RC4 package manifest (Brilliance source `88bd8fc01021b7d2949101301fc69f3c65fa9d92`).
The PE files have no Authenticode signatures. The release tag, wheel pin, SDK's HIP version file
and COMGR version resource establish the vendor origin.
[AMD's wheel index](https://stable.repo.amd.com/rocm/whl-next/rocm-sdk-core/)
[AMD's TheRock 10.1.0 release](https://github.com/ROCm/TheRock/releases/tag/therock-10.1)

| File in `bin/` | Version | Bytes | SHA-256 |
|---|---|---:|---|
| `amdhip64_7.dll` | PE file and product 10.0.3684.0, HIP 7.16.26385 (`7da43026e4` in SDK HIP metadata) | 16,380,416 | `f20e2065cee5889789bf73df0c2d1f4efb481e74ade30488c352fd65ad80ac83` |
| `amd_comgr.dll` | PE file and product 3.5.0.0, commit `d6f6cb691` in the version resource | 123,195,904 | `da9527847c99efd3df99bd46fe24abae7920b04df0f51e7c677abcccd4924e58` |
| `rocm_kpack.dll` | kpack project 0.1.0, wheel 10.1.0. The DLL has no PE file or product version. | 181,248 | `2e9734ffe6f6eefe1ff660a55dc85fffbcd2d66f794fce4878c6d95a55787b48` |

## Inside amdhip64_7.dll

The pinned TheRock Windows recipe selects CLR with PAL, HSA and HIPRTC, static ROCr
(`ROCR_DLL_LOAD=0`) and the `shared/amdgpu-windows-interop` inputs from rocm-systems
`7da43026e4f2ef7600433c38c820732f7a205d22`. The DLL imports kpack and Windows DLLs.
CLR resolves COMGR at run time. The three AMD DLLs each have their own component terms.
[TheRock CLR recipe](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/core/CMakeLists.txt)
[CLR's COMGR loader](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/device/comgrctx.cpp)

| Component | Version or commit | Inside | Licence | Files here (upstream name) | Source |
|---|---|---|---|---|---|
| amdgpu-windows-interop (root terms for the Windows interop bundle) | `7da43026e4f2ef7600433c38c820732f7a205d22` | inputs to amdhip64_7.dll | AMD Software End User License Agreement (LicenseRef-AMD-EULA) | `amdgpu-windows-interop.EULA` (`LICENSE`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop |
| ELFIO (bundled in CLR, `rocclr/elf/elfio`) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `elfio.NOTICE` (the licence comment of `elfio.hpp`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/elf/elfio |
| HIP headers (HIP_COMMON_DIR) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `hip.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/hip |
| HIP runtime (CLR) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `hip-clr.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr |
| MetroHash (AMD's PAL copy) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | Apache-2.0 | `metrohash.LICENSE`, `metrohash.NOTICE` (source attributions) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/shared/metrohash |
| AMD PAL and DevDriver | AMD PAL `3a87fd5d`, static build `ac205ff` | amdhip64_7.dll | MIT text in PAL subtree. Terms for the prebuilt archives remain unresolved. | `pal.LICENSE` and `amdgpu-windows-interop.EULA` | https://github.com/ROCm/rocm-systems/commit/01b4b438400c5ca7b8d5c77c16ce96f9bfe5c631 |
| AMD HSA loader and code reader (`amdhsaloader`, `amdhsacode`) | same rocm-systems commit | amdhip64_7.dll | NCSA source comments | `amd-hsa-loader.NOTICE` | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/sc/HSAIL/ext |
| ELF Tool Chain libelf and bundled headers (`oclelf`) | same rocm-systems commit | amdhip64_7.dll | BSD-family source terms, including uthash | `interop-libelf.NOTICE` | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/hsail-compiler/lib/loaders/elf/utils |
| ROCT thunk (`hsakmt-staticdrm`) | same rocm-systems commit | amdhip64_7.dll | MIT and Nginx BSD-2-Clause notice | `roct-thunk.LICENSE` | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime/libhsakmt |
| AMD Windows kernel-mode interface (`wkmi.lib`) | AMD wkmi `bba4d0323bd9c2ebd86a7975709ac289e6b6007c`, Windows DVC object MD5 `f1a4636bfc6eadb6e9a20f9681bfff1c` | amdhip64_7.dll via ROCr/ROCT | MIT header. No independent binary grant identified for the archive. | `amdgpu-windows-interop.EULA`, `wkmi-header.NOTICE` | https://github.com/ROCm/rocm-systems/commit/289e4a39c6264e7c0d94cc309e2a9efc4915c5e0 |
| AMD RDF and zstd from PAL | public PAL counterpart documented below. Exact archive third-party revisions unresolved. | amdhip64_7.dll | MIT and BSD-3-Clause in the public counterpart | `pal-rdf.NOTICE`, `pal-zstd.LICENSE` | https://github.com/GPUOpen-Drivers/pal/tree/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/imported/rdf |
| ROCclr (part of CLR) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | MIT | `rocclr.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr |
| ROCr hsa-runtime subtree | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | NCSA | `rocr-hsa-runtime.LICENSE` (`LICENSE.md`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime/runtime/hsa-runtime |
| ROCr / HSA runtime (compiled into CLR on Windows) | `7da43026e4f2ef7600433c38c820732f7a205d22` | amdhip64_7.dll | NCSA | `rocr-runtime.LICENSE` (`LICENSE.txt`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime |

### PAL and other link inputs

The PAL CMake target offers 34 DVC-managed AMD Windows Release x64 archives to the linker.
Every pinned pointer matches AMD's import of PAL `3a87fd5d` and static build `ac205ff`. AMD later
reverted the update that would have switched these to PAL `7bd7f8a5` and static build `fdd287d`.
The recipe sets PAL client interface 995 and GPUOpen interface 42.
[PAL link list](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/CMakeLists.txt)
[DVC pointers](https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/lib/Release/x64)
[AMD's revert](https://github.com/ROCm/rocm-systems/commit/54507c1e3e8c3ece61506d7a1572c31f3685581f)

- **PAL platform and services.** `pal.lib`, `palCompilerDeps.lib`, `palUtil.lib`, `addrlib.lib`,
  `vam.lib`, `DriverUtilsService.lib`, `SettingsRpcService2.lib` and `UberTraceService.lib`.
  The DLL carries PAL device and settings strings.
- **DevDriver.** `devdriver.lib`, `ddApi.lib`, `ddCommon.lib`, `ddCore.lib`,
  `ddEventClient.lib`, `ddEventParser.lib`, `ddEventServer.lib`, `ddEventStreamer.lib`,
  `ddNet.lib`, `ddRpcClient.lib`, `ddRpcServer.lib`, `ddRpcShared.lib`, `ddSocket.lib`,
  `ddYaml.lib`, `dd_common.lib`, `dd_crc32.lib` and `dd_settings.lib`. The DLL carries
  DevDriver type data.
- **Corroborated PAL dependencies.** `metrohash.lib`, `amdrdf.lib` and `zstd.lib`. The DLL
  has clusters of the pinned MetroHash64/128 constants, AMD RDF file markers and RDF stream
  type data. The public PAL RDF source uses zstd and the HIP DLL contains zstd frame-magic
  immediates. These observations support inclusion. The archive member list and exact RDF/zstd
  source revisions require upstream build records. The package includes the public PAL
  counterpart's notices.
- **Other PAL inputs.** `pal_lz4.lib`, `mpack.lib`, `cwpack.lib`, `dd_libyaml.lib`,
  `stb_sprintf.lib` and `pal_uuid.lib`. The link recipe identifies these archives.
  The selected object members, upstream versions and exact notice obligations remain open.
  Name-string searches cannot establish that a linker discarded an archive. Candidate texts
  include BSD-2-Clause for LZ4, MIT for MPack, CWPack and libyaml, and MIT or public-domain
  terms for stb_sprintf in the public counterparts. The exact UUID implementation is unverified.
- **Windows HSA and toolchain.** The open part of the link uses the pinned static HSA runtime,
  ROCT and the PAL loader's `amdhsaloader` → `amdhsacode` → `oclelf` chain. The HIP DLL uses
  the Windows static CRT (`/MT`). TheRock supplies header-only SIMDe 0.8.2 to the CLR build,
  whose MIT text is included as `simde.LICENSE`. Source selection and inlining affect which
  header code is emitted.
  [ROCclr PAL recipe](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/cmake/ROCclrPAL.cmake)
  [ROCclr HSA recipe](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/cmake/ROCclrHSA.cmake)

### MetroHash attribution

The complete pinned PAL MetroHash source tree has no upstream `NOTICE` file. Apache-2.0 §4(d)
requires passing through applicable attributions from a supplied NOTICE. The package carries
the Apache text and the pinned source's attribution to J. Andrew Rogers and AMD's modifications
in `metrohash.NOTICE`. Brilliance copies the vendor DLL unchanged. The corresponding archive's
own build metadata and any additional notices have not been established.
[Pinned MetroHash source](https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/shared/metrohash)

## Inside rocm_kpack.dll

| Component | Version or commit | Inside | Licence | Files here (upstream name) | Source |
|---|---|---|---|---|---|
| msgpack-cxx 7.0.0 (header-only) | cpp-7.0.0 | rocm_kpack.dll | BSL-1.0 | `msgpack-cxx.LICENSE` (`LICENSE_1_0.txt`) | https://github.com/msgpack/msgpack-c |
| rocm-kpack | `7da43026e4f2ef7600433c38c820732f7a205d22` | rocm_kpack.dll | MIT | `rocm-kpack.LICENSE` (`LICENSE`) | https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/kpack |
| zstd 1.5.7 (static, TheRock's bundled copy; dual BSD-3-Clause OR GPL-2.0-only, BSD option taken) | v1.5.7 | rocm_kpack.dll | BSD-3-Clause | `zstd.LICENSE` (`LICENSE`) | https://github.com/facebook/zstd |

## Inside amd_comgr.dll

TheRock builds COMGR from AMD's LLVM fork with `COMGR_STATIC_LLVM=ON`. The recipe selects LLVM
(including AMDGPU and SPIR-V), Clang, LLD, and AMD's SPIR-V translator. The DLL carries LLVM,
Clang and translator diagnostics, zlib 1.3.2's embedded copyright strings and zstd code markers.
The compiler recipe also builds AMD's device libraries, and this DLL contains named OCML/OCKL
resources and LLVM bitcode. Its VC++ CRT dependencies are shared. A final vendor link map
would settle the full inventory of object members and embedded compiler resources.
[TheRock compiler recipe](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/compiler/CMakeLists.txt)
[TheRock compiler setup](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/compiler/pre_hook_amd-llvm.cmake)
[COMGR link rules](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/amd/comgr/CMakeLists.txt)

| Component | Licence | Files here |
|---|---|---|
| COMGR 3.5.0, AMD LLVM commit `d6f6cb691863e9258034ab48c1dfce21689d5606` | Apache-2.0 WITH LLVM-exception and legacy NCSA | `amd_comgr.LICENSE.txt` (wheel's complete text) |
| LLVM, Clang and LLD, same commit | Apache-2.0 WITH LLVM-exception and legacy NCSA | `comgr-llvm.LICENSE`, `comgr-clang.LICENSE`, `comgr-lld.LICENSE` |
| LLVM Support, same commit | OpenBSD regex terms including BSD-3-Clause, xxHash BSD-2-Clause, MD5 public domain or permissive fallback, BLAKE3 CC0-1.0 or Apache-2.0 | `comgr-llvm-support.NOTICE`, `comgr-blake3.LICENSE` |
| SPIR-V LLVM Translator `4fd57e737272d1a9077a7ff874961bbc82af2024` and its SPIR-V Headers `575b6512579ebde466ed3dfc04e413439d14d95d` | NCSA and MIT for headers used | `comgr-spirv-translator.LICENSE`, `comgr-spirv-headers.LICENSE`, `comgr-spirv-header.NOTICE` |
| AMD ROCm device libraries, same AMD LLVM commit | NCSA | `comgr-device-libs.LICENSE` |
| TheRock zlib 1.3.2 and zstd 1.5.7 | zlib and BSD-3-Clause | `zlib.LICENSE`, `zstd.LICENSE` |

The translator can optionally link SPIRV-Tools. A COMGR diagnostic identifies a separately
invoked SPIRV-Tools executable. SPIRV-Tools static membership remains unverified.

LLVM's regex implementation derives from Henry Spencer's regex and the OpenBSD adaptation.
The complete attribution and notices are in `comgr-llvm-support.NOTICE`.

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
- Adding an FFmpeg library or encoder to `win\ffmpeg` requires indexing that component's licence.
  The package ships neither `avfilter` nor `avdevice`, and carries no FFmpeg command-line tools.

## Built into radiance.exe: AMD's ADL declarations

`win\os\gpu_win32.cpp` declares the few structures, constants and function types of AMD's ADL
(AMD Display Library) it needs to read each AMD card's board power, GFX activity and junction
temperature for the dashboard. They are taken from the ADL SDK's headers (`adl_sdk.h`,
`adl_defines.h`, `adl_structures.h`; GPUOpen-LibrariesAndSDKs/display-library commit
`84c881cb2136f04287eb3af12de00a9e2867aa9e`), which are MIT: `adl-sdk.LICENSE`, Copyright (c)
2016 - 2022 Advanced Micro Devices, Inc. The package ships no ADL code or DLL: the runtime,
`atiadlxx.dll`, comes with the AMD driver and is loaded at run time from System32. AMD's newer
ADLX is not used; its SDK licence does not allow an open-source engine to carry it.

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

Keep the MIT copyright and permission notices with substantial copies. Reproduce the NCSA and
BSD copyright notices, conditions and disclaimers in the distribution materials. Keep the
Apache-2.0 licence, applicable NOTICE attributions and any required modification notices.
The LLVM exception for portions embedded as a byproduct of compiling other software has
different scope from redistributing AMD's compiler DLL itself. Boost 1.0 exempts copies solely
in machine-executable object code from its text requirement. zlib requires preservation of
its source notice and attribution and marking conditions on the terms in its text. The package
includes those two texts as well.

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

**The ROCm Core SDK DLLs.** AMD built and published the binaries from the TheRock 10.1.0 release.
Brilliance applies no source or binary patches to these DLLs. The source and build recipe are
pinned as follows.

- TheRock, commit `b324869fb3116353ac7eec5e6926aa3d452ac987`:
  https://github.com/ROCm/TheRock/tree/b324869fb3116353ac7eec5e6926aa3d452ac987
- rocm-systems, commit `7da43026e4f2ef7600433c38c820732f7a205d22`:
  https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22
  (`projects/clr`, `projects/hip`, `projects/rocr-runtime`, `shared/kpack`,
  `shared/amdgpu-windows-interop`).
- AMD LLVM, commit `d6f6cb691863e9258034ab48c1dfce21689d5606`:
  https://github.com/ROCm/llvm-project/tree/d6f6cb691863e9258034ab48c1dfce21689d5606
  (`amd/comgr`, `llvm`, `clang`, `lld`, `amd/device-libs`). TheRock also pins the
  [AMD SPIR-V translator](https://github.com/ROCm/SPIRV-LLVM-Translator/tree/4fd57e737272d1a9077a7ff874961bbc82af2024).
- msgpack-cxx `cpp-7.0.0` (https://github.com/msgpack/msgpack-c) and zstd `v1.5.7`
  (https://github.com/facebook/zstd) are TheRock's kpack inputs. TheRock builds zstd and zlib
  1.3.2 as static Windows dependencies. The vendor recipe keeps its Linux source-patching steps
  platform-specific. AMD's PAL MetroHash source carries AMD modifications. The precise delta
  between AMD's prebuilt PAL or wkmi archives and publicly available source is unresolved.

## AMD's Windows HIP distribution terms

**Terms present in the pinned build's sources.** The root of `shared/amdgpu-windows-interop`
carries the AMD Software End User License Agreement, included verbatim in
`amdgpu-windows-interop.EULA`. Sections 1 and 2 cover works incorporating the software and
provide a limited object-code grant for systems or components containing AMD products. Section
2.3 permits distribution of covered Derivative Works to partners and customers under terms at
least as restrictive, subject to any confidentiality agreement. Section 3 restricts distribution
of the Software outside the express grant and preserves upstream copyright, trademark and patent
notices. Section 4 gives applicable third-party terms priority for their own components.
[Pinned interop EULA](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/LICENSE)

**Terms for the complete wheel DLL.** AMD's public wheel carries a licence text for COMGR.
The wheel has no HIP DLL-specific licence or redistribution statement. The PAL subtree's MIT
text and the interop root's EULA are both present in the pinned sources. The available records
leave open which terms AMD applies to the full prebuilt PAL/DevDriver archive set and to
`wkmi.lib`, whether §2.3 covers re-bundling this unmodified vendor DLL in the public Brilliance
ZIP, and which downstream terms AMD requires. Carrying these texts documents their provenance
and preserves component notices.
[ROCm 10.1.0 component licensing](https://rocm.docs.amd.com/en/docs-10.1.0/about/license.html)
[AMD's description of the partial Windows build](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/docs/development/windows_support.md#building-clr-from-partial-sources)

**Windows HIP SDK installer.** AMD's separate installer SDK agreement, Schedule A and §§2–4,
describes incorporation and object-code redistribution for covered materials, only for the AMD
hardware purpose and subject to an enforceable downstream EULA. Section 3 calls for substantially
similar terms to §§4, 5, 8, 10, 12, 15, 16 and 17, plus transfer and duplication limits, reverse
engineering limits, disclaimers and AMD liability limits, export obligations, restricted-purpose
notice and AMD's rights as a third-party beneficiary. AMD's Windows HIP SDK deployment
guidelines direct applications to use the driver-provided HIP runtime and describe application
distribution of HIP SDK libraries. No installer agreement or document reviewed identifies the
RC4 TheRock wheel's HIP DLL by version and hash as covered by an installer redistribution grant.
[AMD HIP SDK installer agreement](https://www.amd.com/content/dam/amd/en/documents/developer/hip-sdk-installer-license-agreement.pdf)
[AMD Windows HIP SDK deployment guidelines](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/deployment-guidelines.html)

## Where each text was fetched

The SHA-256 column identifies the upstream bytes as fetched. Rows marked as compiled notices
give the hash of the file shipped. `llvm-openmp.LICENSE` and `msvc-runtime.NOTICE` come from
Visual Studio Build Tools.

| File | Fetched from | SHA-256 of the upstream file |
|---|---|---|
| `adl-sdk.LICENSE` | https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/display-library/84c881cb2136f04287eb3af12de00a9e2867aa9e/include/adl_sdk.h (whole header; the file here is its leading comment, with LF line ends) | `1d0141f0ff0d8ce11c40b1e37174827f6e0536620365b20d24b55f19adf97d63` |
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

### AMD SDK texts and source notices

| File | Source | SHA-256 |
|---|---|---|
| `amd_comgr.LICENSE.txt` | AMD's pinned wheel, `_rocm_sdk_core/share/doc/amd_comgr/LICENSE.txt`. Text matches [COMGR LICENSE.txt](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/amd/comgr/LICENSE.txt) after line-ending normalization. | `efa88a0f81075b46f0b5eed592d8c575a7014ba402b912da68926766464f9513` |
| `amd-hsa-loader.NOTICE` | Compiled NCSA source comments from [loaders.cpp](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/sc/HSAIL/ext/loader/loaders.cpp) and [executable.cpp](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/sc/HSAIL/ext/loader/executable.cpp), with the files carrying each text identified inside. | `f7b9c53415a0c0db8f8164cf1c2a7fe6c8f08ac99f68772d310d376a891c84a1` |
| `comgr-blake3.LICENSE` | [LLVM BLAKE3 LICENSE](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Support/BLAKE3/LICENSE) | `6a94bedb8b707ed97f6e310d0d015ab14e0683ffa0a612b02958581b9cc9fc0e` |
| `comgr-clang.LICENSE` | [Clang LICENSE.TXT](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/clang/LICENSE.TXT) | `ebcd9bbf783a73d05c53ba4d586b8d5813dcdf3bbec50265860ccc885e606f47` |
| `comgr-device-libs.LICENSE` | [AMD Device Libraries LICENSE.TXT](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/amd/device-libs/LICENSE.TXT) | `a4e8e3ba6e9255928a29d3085f929de402e61413cb0c49276e16b558effa8a2a` |
| `comgr-lld.LICENSE` | [LLD LICENSE.TXT](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/lld/LICENSE.TXT) | `f7891568956e34643eb6a0db1462db30820d40d7266e2a78063f2fe233ece5a0` |
| `comgr-llvm-support.NOTICE` | Compiled notice with [COPYRIGHT.regex](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Support/COPYRIGHT.regex), [xxhash.cpp](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Support/xxhash.cpp), [xxhash.h](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/include/llvm/Support/xxhash.h) and [MD5.cpp](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Support/MD5.cpp). | `647e581d606cdb16d94afda72374158edf70a9ff7c1d74e3f4584ac66f4a0317` |
| `comgr-llvm.LICENSE` | [LLVM LICENSE.TXT](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/LICENSE.TXT) | `8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee` |
| `comgr-spirv-header.NOTICE` | Compiled notice with copyright and licence lines from [spirv.hpp](https://github.com/KhronosGroup/SPIRV-Headers/blob/575b6512579ebde466ed3dfc04e413439d14d95d/include/spirv/unified1/spirv.hpp) | `ed04b63a0421f95e60835ae733a96cc358f905ef00e232b9e56899db387d7040` |
| `comgr-spirv-headers.LICENSE` | [SPIR-V Headers LICENSE](https://github.com/KhronosGroup/SPIRV-Headers/blob/575b6512579ebde466ed3dfc04e413439d14d95d/LICENSE) | `ea43b1de38a6f90c488800d66dec1ed671e68cda530266bc96951fb5b6307613` |
| `comgr-spirv-translator.LICENSE` | [AMD SPIR-V LLVM Translator LICENSE.TXT](https://github.com/ROCm/SPIRV-LLVM-Translator/blob/4fd57e737272d1a9077a7ff874961bbc82af2024/LICENSE.TXT) | `e3bc36440fc927c62d5cc24efeefe225a14d4e34ffeb0c92e430625cce9ee444` |
| `interop-libelf.NOTICE` | Compiled notices from [libelf and bundled headers](https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/hsail-compiler/lib/loaders/elf/utils), with source paths inside. | `e34d5b8d02c8ecd890da8e905df37f6cd3df15df65eeb9d5d8593611703e7b72` |
| `metrohash.NOTICE` | Compiled source attributions as carried in [MetroHash64](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/shared/metrohash/src/metrohash64.cpp) and [MetroHash128](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/pal/shared/metrohash/src/metrohash128.cpp). | `b1cdb820e7b136740752a1a7682719d1d94c6f05ef22f5187a89535fb5ba0ec6` |
| `pal-rdf.NOTICE` | Compiled notice with the MIT block from [amdrdf.cpp](https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/imported/rdf/rdf/src/amdrdf.cpp), the public PAL counterpart. | `c4b5d53382396a2c7077048c873a8facf3f0e157d5d0c77b3ea7160d707e702e` |
| `pal-zstd.LICENSE` | [Public PAL RDF's zstd LICENSE](https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/imported/rdf/imported/zstd/LICENSE) | `8b7bc50ec3ecee27224e17d4d316a939fa2ddda9c88e0ddc2f059ce432edbfb9` |
| `roct-thunk.LICENSE` | [ROCT Thunk Interface LICENSE.md](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime/libhsakmt/LICENSE.md) | `f45a5bc0bc68dd9a00e9609097788b1cbda79726fee02549cb0c04b90190b419` |
| `simde.LICENSE` | [SIMDe v0.8.2 COPYING](https://github.com/simd-everywhere/simde/blob/v0.8.2/COPYING) | `75ce0a8c5a7bff42fbd6f8f3b469b518132132c4f4948fb8d02b1ccb5d0df9d8` |
| `wkmi-header.NOTICE` | Compiled notice with the MIT SPDX comment from [wkmi.h](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/shared/amdgpu-windows-interop/wkmi/wkmi.h). | `fbe7cc6776daf845a5ec37aa029d85b524fb3afa8c30eebd2522f21ddae51dc1` |
