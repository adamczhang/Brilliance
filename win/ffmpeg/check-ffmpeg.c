/* check-ffmpeg.c: checks a Brilliance FFmpeg build from its DLLs alone (win/ffmpeg/README.md).
 *
 *   check-ffmpeg <media-dir> <out-dir> [--no-h264-mf]
 *
 * 1. Prints the configure line and licence the DLLs report, and fails unless the licence is
 *    LGPL-2.1-or-later.
 * 2. Lists every encoder the DLLs register, and fails if there is any but png, mjpeg, libwebp
 *    (/v1/images), h264_amf, h264_mf and aac (the /v1/videos writer, core/mm/mp4.cpp: its video
 *    and its soundtrack). So the only H.264 encoders are the wrappers over AMD's (AMF) and
 *    Windows' (Media Foundation) licensed encoders: libopenh264, libx264, every other H.264
 *    encoder and every other video encoder fail the check, and so does any audio encoder but
 *    the native aac. Fails on any muxer but mp4 (and the mov muxer it is built on), on any
 *    protocol, and on the AMF decoders.
 * 3. Decodes every file in <media-dir> the way the engine does (core/mm/media.cpp): from memory
 *    through a custom AVIO, an empty protocol whitelist and the engine's demuxer whitelist; video
 *    and pictures through swscale to RGB24, audio through swresample to 16 kHz mono float.
 * 4. Encodes a picture as PNG, JPEG and WebP (the /v1/images path), writes them to <out-dir>,
 *    and decodes each back; the PNG must come back pixel-exact.
 * 5. Writes an MP4 the way Mp4Writer does (core/mm/mp4.cpp): RGB24 frames through swscale to
 *    YUV420P BT.709, h264_mf through Windows' software H.264 MFT (CPU), an aac soundtrack, the mp4
 *    muxer into memory with +faststart; then decodes the file back from memory. Reports whether
 *    AMD's AMF runtime (amfrt64.dll) is installed; h264_amf is not opened (it needs the GPU).
 *
 * CPU only. Exit 0 when every check passes. */
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static int failures = 0;
#define FAIL(...) do { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } while (0)

/* The engine's demuxer whitelists: kFormats and kAudioFormats in core/mm/media.cpp. */
static const char* kWhitelist =
    "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts,flv,ogg,gif,apng,image2pipe,png_pipe,"
    "jpeg_pipe,jpegls_pipe,jpegxl_pipe,webp_pipe,bmp_pipe,tiff_pipe,qoi_pipe,pgm_pipe,ppm_pipe,"
    "pbm_pipe,pam_pipe,gif_pipe,avif,heif,wav,w64,mp3,flac,aac,aiff";

typedef struct { const uint8_t* p; int64_t n, pos; } Mem;

static int mem_read(void* o, uint8_t* buf, int sz) {
    Mem* m = (Mem*)o;
    int64_t left = m->n - m->pos;
    if (left <= 0) return AVERROR_EOF;
    if (sz > left) sz = (int)left;
    memcpy(buf, m->p + m->pos, sz);
    m->pos += sz;
    return sz;
}
static int64_t mem_seek(void* o, int64_t off, int whence) {
    Mem* m = (Mem*)o;
    if (whence & AVSEEK_SIZE) return m->n;
    whence &= ~AVSEEK_FORCE;
    int64_t np = whence == SEEK_SET ? off : whence == SEEK_CUR ? m->pos + off : m->n + off;
    if (np < 0 || np > m->n) return -1;
    m->pos = np;
    return np;
}

static uint8_t* slurp(const char* path, int64_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    _fseeki64(f, 0, SEEK_END);
    *n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    uint8_t* b = (uint8_t*)malloc((size_t)*n + 1);
    if (b && fread(b, 1, (size_t)*n, f) != (size_t)*n) { free(b); b = NULL; }
    fclose(f);
    return b;
}

/* Decode one file from memory. For a picture or video, rgb (if not NULL) gets the first frame as
 * RGB24 (caller frees). Returns frames (video) or samples (audio) decoded, or -1. */
static int64_t decode_mem(const char* label, const uint8_t* data, int64_t n, uint8_t** rgb,
                          int* w, int* h) {
    Mem m = { data, n, 0 };
    AVFormatContext* fmt = NULL;
    AVCodecContext* dec = NULL;
    AVFrame* fr = av_frame_alloc();
    AVPacket* pk = av_packet_alloc();
    SwrContext* swr = NULL;
    struct SwsContext* sws = NULL;
    int64_t count = -1;
    unsigned char* iobuf = (unsigned char*)av_malloc(65536);
    AVIOContext* io = avio_alloc_context(iobuf, 65536, 0, &m, mem_read, NULL, mem_seek);
    AVDictionary* opt = NULL;
    av_dict_set(&opt, "protocol_whitelist", "", 0);
    av_dict_set(&opt, "format_whitelist", kWhitelist, 0);
    fmt = avformat_alloc_context();
    fmt->pb = io;
    fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
    int r = avformat_open_input(&fmt, NULL, NULL, &opt);
    av_dict_free(&opt);
    if (r < 0) { FAIL("%s: open (%s)", label, av_err2str(r)); fmt = NULL; goto done; }
    if ((r = avformat_find_stream_info(fmt, NULL)) < 0) { FAIL("%s: stream info (%s)", label, av_err2str(r)); goto done; }
    const AVCodec* codec = NULL;
    int si = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    int audio = 0;
    if (si < 0) { si = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0); audio = 1; }
    if (si < 0 || !codec) { FAIL("%s: no decodable stream (%s)", label, av_err2str(si)); goto done; }
    dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, fmt->streams[si]->codecpar);
    if ((r = avcodec_open2(dec, codec, NULL)) < 0) { FAIL("%s: decoder %s (%s)", label, codec->name, av_err2str(r)); goto done; }
    count = 0;
    int eof = 0;
    while (!eof) {
        r = av_read_frame(fmt, pk);
        if (r < 0) { eof = 1; avcodec_send_packet(dec, NULL); }
        else if (pk->stream_index != si) { av_packet_unref(pk); continue; }
        else { avcodec_send_packet(dec, pk); av_packet_unref(pk); }
        while (avcodec_receive_frame(dec, fr) == 0) {
            if (audio) {
                if (!swr) {
                    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
                    if (swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_FLT, 16000, &fr->ch_layout,
                                            (enum AVSampleFormat)fr->format, fr->sample_rate, 0, NULL) < 0 ||
                        swr_init(swr) < 0) { FAIL("%s: swresample", label); count = -1; goto done; }
                }
                float buf[16384];
                uint8_t* outp[1] = { (uint8_t*)buf };
                int got = swr_convert(swr, outp, 16384, (const uint8_t**)fr->extended_data, fr->nb_samples);
                if (got < 0) { FAIL("%s: swr_convert", label); count = -1; goto done; }
                count += got;
            } else {
                if (count == 0) {
                    *w = fr->width; *h = fr->height;
                    sws = sws_getContext(fr->width, fr->height, (enum AVPixelFormat)fr->format, fr->width,
                                         fr->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
                    if (!sws) { FAIL("%s: swscale from %s", label, av_get_pix_fmt_name(fr->format)); count = -1; goto done; }
                    uint8_t* px = (uint8_t*)malloc((size_t)fr->width * fr->height * 3);
                    uint8_t* dst[1] = { px };
                    int ls[1] = { fr->width * 3 };
                    sws_scale(sws, (const uint8_t* const*)fr->data, fr->linesize, 0, fr->height, dst, ls);
                    if (rgb) *rgb = px; else free(px);
                }
                count++;
            }
            av_frame_unref(fr);
        }
    }
    if (audio) {
        float buf[16384];
        uint8_t* outp[1] = { (uint8_t*)buf };
        int got = swr ? swr_convert(swr, outp, 16384, NULL, 0) : 0;
        if (got > 0) count += got;
        printf("  %-20s %-10s %-12s %-12s %8lld samples at 16 kHz mono\n", label, fmt->iformat->name,
               codec->name, av_get_sample_fmt_name(dec->sample_fmt), (long long)count);
    } else {
        printf("  %-20s %-10s %-12s %-12s %4dx%-4d %lld frame(s)\n", label, fmt->iformat->name, codec->name,
               av_get_pix_fmt_name(dec->pix_fmt), *w, *h, (long long)count);
    }
    if (count <= 0) { FAIL("%s: nothing decoded", label); count = -1; }
done:
    sws_freeContext(sws);
    swr_free(&swr);
    avcodec_free_context(&dec);
    if (fmt) avformat_close_input(&fmt);
    if (io) { av_freep(&io->buffer); avio_context_free(&io); }
    av_frame_free(&fr);
    av_packet_free(&pk);
    return count;
}

/* One picture through one encoder, as core/mm/media.cpp encode_image does. */
static int encode_one(const uint8_t* rgb, int w, int h, const char* name, enum AVPixelFormat pf,
                      const char* out_path, uint8_t** file, int64_t* n) {
    const AVCodec* c = avcodec_find_encoder_by_name(name);
    if (!c) { FAIL("no %s encoder", name); return -1; }
    AVCodecContext* cx = avcodec_alloc_context3(c);
    cx->width = w; cx->height = h; cx->pix_fmt = pf; cx->time_base = (AVRational){ 1, 25 };
    if (pf == AV_PIX_FMT_YUVJ420P) {
        cx->color_range = AVCOL_RANGE_JPEG;
        cx->flags |= AV_CODEC_FLAG_QSCALE;
        cx->global_quality = FF_QP2LAMBDA * 4;
    }
    if (!strcmp(name, "png")) { cx->compression_level = 1; av_opt_set(cx->priv_data, "pred", "mixed", 0); }
    if (!strcmp(name, "libwebp")) av_opt_set_double(cx->priv_data, "quality", 90.0, 0);
    int r = avcodec_open2(cx, c, NULL);
    if (r < 0) { FAIL("%s open (%s)", name, av_err2str(r)); avcodec_free_context(&cx); return -1; }
    AVFrame* fr = av_frame_alloc();
    fr->format = pf; fr->width = w; fr->height = h;
    av_frame_get_buffer(fr, 0);
    struct SwsContext* sws = sws_getContext(w, h, AV_PIX_FMT_RGB24, w, h, pf, SWS_BILINEAR | SWS_ACCURATE_RND,
                                            NULL, NULL, NULL);
    const uint8_t* src[1] = { rgb };
    int sl[1] = { w * 3 };
    sws_scale(sws, src, sl, 0, h, fr->data, fr->linesize);
    AVPacket* pk = av_packet_alloc();
    avcodec_send_frame(cx, fr);
    avcodec_send_frame(cx, NULL);
    r = avcodec_receive_packet(cx, pk);
    int ok = r == 0 && pk->size > 0;
    if (ok) {
        *n = pk->size;
        *file = (uint8_t*)malloc(pk->size);
        memcpy(*file, pk->data, pk->size);
        FILE* f = fopen(out_path, "wb");
        if (f) { fwrite(pk->data, 1, pk->size, f); fclose(f); }
        printf("  encode %-8s -> %-28s %7d bytes\n", name, strrchr(out_path, '\\') ? strrchr(out_path, '\\') + 1 : out_path, pk->size);
    } else {
        FAIL("%s encode (%s)", name, av_err2str(r));
    }
    av_packet_free(&pk);
    sws_freeContext(sws);
    av_frame_free(&fr);
    avcodec_free_context(&cx);
    return ok ? 0 : -1;
}

/* ---- An MP4 the way Mp4Writer (core/mm/mp4.cpp) writes one, into memory. */
typedef struct { uint8_t* p; int64_t n, cap, pos; } OutMem;

static int out_write(void* o, const uint8_t* buf, int n) {
    OutMem* m = (OutMem*)o;
    int64_t end = m->pos + n;
    if (end > m->cap) {
        int64_t cap = m->cap ? m->cap : 65536;
        while (cap < end) cap *= 2;
        m->p = (uint8_t*)realloc(m->p, (size_t)cap);
        m->cap = cap;
    }
    if (end > m->n) m->n = end;
    memcpy(m->p + m->pos, buf, n);
    m->pos = end;
    return n;
}
static int64_t out_seek(void* o, int64_t off, int whence) {
    OutMem* m = (OutMem*)o;
    if (whence & AVSEEK_SIZE) return m->n;
    whence &= ~AVSEEK_FORCE;
    int64_t np = whence == SEEK_SET ? off : whence == SEEK_CUR ? m->pos + off : m->n + off;
    if (np < 0) return AVERROR(EINVAL);
    m->pos = np;
    return np;
}
/* +faststart's second pass reads the output back through io_open on the output's own URL. */
static int out_open_reader(AVFormatContext* s, AVIOContext** pb, const char* url, int flags, AVDictionary** o) {
    (void)url; (void)o;
    if (!(flags & AVIO_FLAG_READ) || (flags & AVIO_FLAG_WRITE)) return AVERROR(EINVAL);
    OutMem* m = (OutMem*)s->opaque;
    Mem* r = (Mem*)calloc(1, sizeof(Mem));
    r->p = m->p; r->n = m->n; r->pos = 0;
    uint8_t* buf = (uint8_t*)av_malloc(65536);
    *pb = avio_alloc_context(buf, 65536, 0, r, mem_read, NULL, mem_seek);
    return *pb ? 0 : AVERROR(ENOMEM);
}
static int out_close_reader(AVFormatContext* s, AVIOContext* pb) {
    (void)s;
    if (!pb) return 0;
    free(pb->opaque);
    av_freep(&pb->buffer);
    avio_context_free(&pb);
    return 0;
}

static int drain(AVCodecContext* c, AVStream* st, AVFormatContext* oc, AVPacket* pk, int* packets) {
    int r;
    while ((r = avcodec_receive_packet(c, pk)) == 0) {
        /* h264_mf hands out packets with no duration. The mov muxer then ends the edit list at
         * the last frame's start and marks that frame discard, so a player shows one frame fewer.
         * One frame's duration (the codec time base is 1/fps) puts it back. Mp4Writer needs the
         * same line when it moves to h264_mf. */
        if (pk->duration == 0 && c->codec_type == AVMEDIA_TYPE_VIDEO) pk->duration = 1;
        av_packet_rescale_ts(pk, c->time_base, st->time_base);
        pk->stream_index = st->index;
        (*packets)++;
        if ((r = av_interleaved_write_frame(oc, pk)) < 0) return r;
    }
    return r == AVERROR(EAGAIN) || r == AVERROR_EOF ? 0 : r;
}

/* RGB24 frames -> swscale (YUV420P, limited-range BT.709) -> h264_mf on Windows' software H.264
 * MFT (hw_encoding 0: no GPU); a 440 Hz tone -> aac; both into the mp4 muxer in memory with
 * +faststart. Then the file is decoded back from memory. h264_amf would sit where h264_mf is,
 * but it runs on an AMD GPU through the driver's AMF runtime, so it is not opened here. */
static void mp4_writer_check(const char* outdir) {
    const int W = 128, H = 96, N = 25, RATE = 48000, CH = 2;
    OutMem m = { 0 };
    AVFormatContext* oc = NULL;
    AVCodecContext *vc = NULL, *ac = NULL;
    AVFrame *vf = av_frame_alloc(), *af = av_frame_alloc();
    AVPacket* pk = av_packet_alloc();
    struct SwsContext* sws = NULL;
    uint8_t* rgb = (uint8_t*)malloc((size_t)W * H * 3);
    int r, vpk = 0, apk = 0;
    const AVCodec* venc = avcodec_find_encoder_by_name("h264_mf");
    const AVCodec* aenc = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!venc || !aenc) { FAIL("h264_mf or aac is missing"); goto done; }
    if (avformat_alloc_output_context2(&oc, NULL, "mp4", NULL) < 0 || !oc) { FAIL("no mp4 muxer"); goto done; }
    oc->url = av_strdup("check.mp4");
    oc->opaque = &m;
    oc->io_open = out_open_reader;
    oc->io_close2 = out_close_reader;
    oc->pb = avio_alloc_context((unsigned char*)av_malloc(65536), 65536, 1, &m, NULL, out_write, out_seek);

    AVStream* vs = avformat_new_stream(oc, NULL);
    vc = avcodec_alloc_context3(venc);
    vc->width = W; vc->height = H; vc->pix_fmt = AV_PIX_FMT_YUV420P;
    vc->time_base = (AVRational){ 1, 25 }; vc->framerate = (AVRational){ 25, 1 };
    vc->gop_size = 50; vc->bit_rate = (int64_t)(W * H * 25 * 0.2);
    vc->colorspace = AVCOL_SPC_BT709; vc->color_primaries = AVCOL_PRI_BT709;
    vc->color_trc = AVCOL_TRC_BT709; vc->color_range = AVCOL_RANGE_MPEG;
    av_opt_set_int(vc->priv_data, "hw_encoding", 0, 0);
    if (oc->oformat->flags & AVFMT_GLOBALHEADER) vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if ((r = avcodec_open2(vc, venc, NULL)) < 0) { FAIL("h264_mf would not open (%s)", av_err2str(r)); goto done; }
    avcodec_parameters_from_context(vs->codecpar, vc);
    vs->time_base = vc->time_base;

    AVStream* as = avformat_new_stream(oc, NULL);
    ac = avcodec_alloc_context3(aenc);
    ac->sample_fmt = AV_SAMPLE_FMT_FLTP; ac->sample_rate = RATE; ac->bit_rate = 96000 * CH;
    av_channel_layout_default(&ac->ch_layout, CH);
    ac->time_base = (AVRational){ 1, RATE };
    if (oc->oformat->flags & AVFMT_GLOBALHEADER) ac->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if ((r = avcodec_open2(ac, aenc, NULL)) < 0) { FAIL("aac would not open (%s)", av_err2str(r)); goto done; }
    avcodec_parameters_from_context(as->codecpar, ac);
    as->time_base = ac->time_base;

    AVDictionary* opts = NULL;
    av_dict_set(&opts, "movflags", "+faststart", 0);
    r = avformat_write_header(oc, &opts);
    av_dict_free(&opts);
    if (r < 0) { FAIL("mp4 header (%s)", av_err2str(r)); goto done; }

    vf->format = AV_PIX_FMT_YUV420P; vf->width = W; vf->height = H;
    av_frame_get_buffer(vf, 0);
    sws = sws_getContext(W, H, AV_PIX_FMT_RGB24, W, H, AV_PIX_FMT_YUV420P, SWS_BILINEAR, NULL, NULL, NULL);
    if (!sws) { FAIL("swscale RGB24 -> YUV420P"); goto done; }
    sws_setColorspaceDetails(sws, sws_getCoefficients(SWS_CS_DEFAULT), 1, sws_getCoefficients(SWS_CS_ITU709), 0,
                             0, 1 << 16, 1 << 16);
    for (int i = 0; i < N; i++) {
        for (int p = 0; p < W * H; p++) {
            rgb[3 * p] = (uint8_t)(p % W * 2 + 4 * i); rgb[3 * p + 1] = (uint8_t)(p / W * 2); rgb[3 * p + 2] = 96;
        }
        av_frame_make_writable(vf);
        const uint8_t* src[1] = { rgb };
        int sl[1] = { W * 3 };
        sws_scale(sws, src, sl, 0, H, vf->data, vf->linesize);
        vf->pts = i;
        if ((r = avcodec_send_frame(vc, vf)) < 0) { FAIL("h264_mf frame (%s)", av_err2str(r)); goto done; }
        if ((r = drain(vc, vs, oc, pk, &vpk)) < 0) { FAIL("writing video (%s)", av_err2str(r)); goto done; }
    }
    af->format = AV_SAMPLE_FMT_FLTP; af->sample_rate = RATE; af->nb_samples = ac->frame_size;
    av_channel_layout_copy(&af->ch_layout, &ac->ch_layout);
    av_frame_get_buffer(af, 0);
    for (int64_t t = 0; t < RATE; t += ac->frame_size) {   /* one second */
        av_frame_make_writable(af);
        for (int c = 0; c < CH; c++)
            for (int k = 0; k < ac->frame_size; k++)
                ((float*)af->data[c])[k] = 0.25f * (float)sin(2 * 3.14159265358979 * 440 * (double)(t + k) / RATE);
        af->pts = t;
        if ((r = avcodec_send_frame(ac, af)) < 0) { FAIL("aac frame (%s)", av_err2str(r)); goto done; }
        if ((r = drain(ac, as, oc, pk, &apk)) < 0) { FAIL("writing audio (%s)", av_err2str(r)); goto done; }
    }
    avcodec_send_frame(ac, NULL);
    if ((r = drain(ac, as, oc, pk, &apk)) < 0) { FAIL("writing audio (%s)", av_err2str(r)); goto done; }
    avcodec_send_frame(vc, NULL);
    if ((r = drain(vc, vs, oc, pk, &vpk)) < 0) { FAIL("writing video (%s)", av_err2str(r)); goto done; }
    if ((r = av_write_trailer(oc)) < 0) { FAIL("mp4 trailer (%s)", av_err2str(r)); goto done; }
    avio_flush(oc->pb);
    printf("  mp4: h264_mf %d packets, aac %d packets, %lld bytes, +faststart\n", vpk, apk, (long long)m.n);

    /* moov before mdat: what +faststart promises. */
    const uint8_t* moov = NULL;
    const uint8_t* mdat = NULL;
    for (int64_t i = 4; i + 4 <= m.n; i++) {
        if (!moov && !memcmp(m.p + i, "moov", 4)) moov = m.p + i;
        if (!mdat && !memcmp(m.p + i, "mdat", 4)) mdat = m.p + i;
    }
    if (!moov || !mdat || moov > mdat) FAIL("mp4 is not faststart (moov after mdat)");
    char op[MAX_PATH];
    snprintf(op, sizeof op, "%s\\check-h264_mf-aac.mp4", outdir);
    FILE* f = fopen(op, "wb");
    if (f) { fwrite(m.p, 1, (size_t)m.n, f); fclose(f); }
    int w = 0, h = 0;
    int64_t frames = decode_mem("check-h264_mf-aac.mp4", m.p, m.n, NULL, &w, &h);
    if (frames != N || w != W || h != H) FAIL("the mp4's video decoded to %lld frames of %dx%d", (long long)frames, w, h);
done:
    sws_freeContext(sws);
    avcodec_free_context(&vc);
    avcodec_free_context(&ac);
    if (oc) {
        if (oc->pb) { av_freep(&oc->pb->buffer); avio_context_free(&oc->pb); }
        avformat_free_context(oc);
    }
    av_frame_free(&vf);
    av_frame_free(&af);
    av_packet_free(&pk);
    free(rgb);
    free(m.p);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: check-ffmpeg <media-dir> <out-dir>\n"); return 2; }
    av_log_set_level(AV_LOG_ERROR);
    const char* media = argv[1];
    const char* outdir = argv[2];
    CreateDirectoryA(outdir, NULL);

    printf("== versions and licence\n");
    printf("  avutil %s, avcodec %u, avformat %u, swscale %u, swresample %u\n", av_version_info(),
           avcodec_version(), avformat_version(), swscale_version(), swresample_version());
    printf("  configuration: %s\n", avcodec_configuration());
    const char* lic[5] = { avutil_license(), avcodec_license(), avformat_license(), swscale_license(),
                           swresample_license() };
    for (int i = 0; i < 5; i++) {
        printf("  licence[%d]: %s\n", i, lic[i]);
        if (strcmp(lic[i], "LGPL version 2.1 or later") != 0) FAIL("licence is '%s', not LGPL-2.1-or-later", lic[i]);
    }
    if (strstr(avcodec_configuration(), "--enable-gpl") || strstr(avcodec_configuration(), "--enable-nonfree"))
        FAIL("configured with gpl or nonfree");
    if (strstr(avcodec_configuration(), "--enable-libopenh264") || strstr(avcodec_configuration(), "--enable-libx264"))
        FAIL("configured with an H.264 encoder library");

    /* The picture encoders, and exactly two H.264 encoders, both wrappers over an encoder its maker
     * licenses: AMD's through AMF (the driver's amfrt64.dll) and Windows Media Foundation's. */
    printf("== encoders (every one registered)\n");
    const char* allowed[] = { "png", "mjpeg", "libwebp", "h264_amf", "h264_mf", "aac" };
    const int nallowed = (int)(sizeof allowed / sizeof *allowed);
    void* it = NULL;
    const AVCodec* c;
    int nenc = 0, ndec = 0, nh264 = 0;
    while ((c = av_codec_iterate(&it))) {
        if (!av_codec_is_encoder(c)) { ndec++; continue; }
        nenc++;
        printf("  %-10s %-6s %-6s %s\n", c->name, av_get_media_type_string(c->type),
               (c->capabilities & AV_CODEC_CAP_HARDWARE) ? "hw" : "",
               c->long_name ? c->long_name : "");
        int ok = 0;
        for (int i = 0; i < nallowed; i++) ok |= !strcmp(c->name, allowed[i]);
        if (!ok) FAIL("encoder %s is not in the allowed set (png, mjpeg, libwebp, h264_amf, h264_mf, aac)", c->name);
        if (c->id == AV_CODEC_ID_H264) nh264++;
        if (c->id == AV_CODEC_ID_HEVC || c->id == AV_CODEC_ID_AV1 || c->id == AV_CODEC_ID_VP9)
            FAIL("encoder %s encodes HEVC/AV1/VP9", c->name);
        if (c->type == AVMEDIA_TYPE_AUDIO && strcmp(c->name, "aac")) FAIL("audio encoder %s", c->name);
    }
    if (nh264 != 2 || !avcodec_find_encoder_by_name("h264_amf") || !avcodec_find_encoder_by_name("h264_mf"))
        FAIL("H.264 encoders are not exactly h264_amf and h264_mf (%d found)", nh264);
    const AVCodec* h264def = avcodec_find_encoder(AV_CODEC_ID_H264);
    printf("  avcodec_find_encoder(H264) -> %s\n", h264def ? h264def->name : "(none)");
    const char* banned[] = { "libopenh264", "libx264", "libx264rgb", "h264_d3d12va", "h264_nvenc", "h264_qsv",
                             "h264_vaapi", "h264_vulkan", "h264_v4l2m2m", "h264_videotoolbox", "libx265",
                             "hevc_amf", "hevc_mf", "av1_amf", "av1_mf", "libvpx-vp9", "libsvtav1",
                             "libaom-av1", "aac_mf", "mp3_mf", "ac3_mf", "libfdk_aac", "libopus",
                             "libmp3lame", "mpeg4", "mpeg2video", "libvpx", "ffv1", "prores" };
    for (int i = 0; i < (int)(sizeof banned / sizeof *banned); i++)
        if (avcodec_find_encoder_by_name(banned[i])) FAIL("encoder %s present", banned[i]);
    const char* amfdec[] = { "h264_amf", "hevc_amf", "av1_amf", "vp9_amf" };
    for (int i = 0; i < 4; i++)
        if (avcodec_find_decoder_by_name(amfdec[i])) FAIL("AMF decoder %s present", amfdec[i]);
    printf("  %d encoder(s), %d of them H.264; %d decoder(s)\n", nenc, nh264, ndec);

    printf("== decoders the engine relies on\n");
    const char* want[] = { "h264", "hevc", "vp8", "vp9", "libdav1d", "mpeg4", "mpeg1video", "mpeg2video", "mjpeg",
                           "theora", "h263", "flv", "prores", "png", "apng", "gif", "webp", "bmp", "tiff", "qoi",
                           "pgm", "ppm", "pbm", "pam", "jpegls", "mp3float", "mp3", "flac", "vorbis", "opus",
                           "aac", "alac", "pcm_s16le", "pcm_s16be", "pcm_f32le", "adpcm_ms", "adpcm_ima_wav" };
    for (int i = 0; i < (int)(sizeof want / sizeof *want); i++)
        if (!avcodec_find_decoder_by_name(want[i])) FAIL("decoder %s missing", want[i]);
    const AVCodec* av1 = avcodec_find_decoder(AV_CODEC_ID_AV1);
    printf("  AV1 resolves to: %s\n", av1 ? av1->name : "(none)");
    if (!av1 || strcmp(av1->name, "libdav1d")) FAIL("AV1 does not resolve to libdav1d");
    printf("  JPEG XL decoder: %s\n", avcodec_find_decoder(AV_CODEC_ID_JPEGXL) ? "yes" : "no (no libjxl)");

    printf("== muxers and protocols\n");
    int nmux = 0, nprot = 0;
    it = NULL;
    const AVOutputFormat* of;
    while ((of = av_muxer_iterate(&it))) {
        nmux++;
        printf("  muxer %s\n", of->name);
        if (strcmp(of->name, "mp4") && strcmp(of->name, "mov")) FAIL("muxer %s is not mp4 or mov", of->name);
    }
    if (!av_guess_format("mp4", NULL, NULL)) FAIL("no mp4 muxer");
    it = NULL;
    const char* pn;
    while ((pn = avio_enum_protocols(&it, 0))) { nprot++; printf("  input protocol %s\n", pn); }
    it = NULL;
    while ((pn = avio_enum_protocols(&it, 1))) { nprot++; printf("  output protocol %s\n", pn); }
    printf("  %d muxer(s), %d protocol(s)\n", nmux, nprot);
    if (nprot) FAIL("%d protocols registered", nprot);

    printf("== decode (memory AVIO, engine whitelist)\n");
    char pat[MAX_PATH];
    snprintf(pat, sizeof pat, "%s\\*", media);
    WIN32_FIND_DATAA fd;
    HANDLE hf = FindFirstFileA(pat, &fd);
    uint8_t* quad = NULL;
    int qw = 0, qh = 0;
    int ndecoded = 0;
    if (hf != INVALID_HANDLE_VALUE) {
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            const char* ext = strrchr(fd.cFileName, '.');
            if (!ext || !_stricmp(ext, ".md") || !_stricmp(ext, ".txt")) continue;
            char path[MAX_PATH];
            snprintf(path, sizeof path, "%s\\%s", media, fd.cFileName);
            int64_t n = 0;
            uint8_t* b = slurp(path, &n);
            if (!b) { FAIL("read %s", path); continue; }
            uint8_t* rgb = NULL;
            int w = 0, h = 0;
            int is_quad = !strcmp(fd.cFileName, "quad.png");
            if (decode_mem(fd.cFileName, b, n, is_quad ? &rgb : NULL, &w, &h) > 0) ndecoded++;
            if (is_quad && rgb) { quad = rgb; qw = w; qh = h; }
            free(b);
        } while (FindNextFileA(hf, &fd));
        FindClose(hf);
    }
    printf("  %d file(s) decoded\n", ndecoded);

    printf("== encode (the /v1/images formats), then decode back\n");
    if (!quad) {
        FAIL("quad.png did not decode; no picture to encode");
    } else {
        struct { const char* enc; enum AVPixelFormat pf; const char* file; } E[] = {
            { "png", AV_PIX_FMT_RGB24, "quad-out.png" },
            { "mjpeg", AV_PIX_FMT_YUVJ420P, "quad-out.jpg" },
            { "libwebp", AV_PIX_FMT_BGRA, "quad-out.webp" },
        };
        for (int i = 0; i < 3; i++) {
            char op[MAX_PATH];
            snprintf(op, sizeof op, "%s\\%s", outdir, E[i].file);
            uint8_t* file = NULL;
            int64_t n = 0;
            if (encode_one(quad, qw, qh, E[i].enc, E[i].pf, op, &file, &n) < 0) continue;
            uint8_t* back = NULL;
            int w = 0, h = 0;
            if (decode_mem(E[i].file, file, n, &back, &w, &h) > 0) {
                if (w != qw || h != qh) FAIL("%s came back %dx%d, not %dx%d", E[i].file, w, h, qw, qh);
                if (!strcmp(E[i].enc, "png") && back && memcmp(back, quad, (size_t)qw * qh * 3))
                    FAIL("png round trip is not pixel-exact");
                if (!strcmp(E[i].enc, "png") && back && !memcmp(back, quad, (size_t)qw * qh * 3))
                    printf("  png round trip pixel-exact\n");
            }
            free(back);
            free(file);
        }
        free(quad);
    }

    printf("== the video writer: H.264 through a licensed encoder, aac, mp4\n");
    if (argc > 3 && !strcmp(argv[3], "--no-h264-mf")) {
        printf("  MP4 check skipped (--no-h264-mf)\n");
    } else {
        mp4_writer_check(outdir);
    }
    char amfrt[MAX_PATH];
    GetSystemDirectoryA(amfrt, MAX_PATH);
    strcat_s(amfrt, MAX_PATH, "\\amfrt64.dll");
    printf("  h264_amf: registered; not opened here (it needs an AMD GPU). AMD's runtime %s: %s\n", amfrt,
           GetFileAttributesA(amfrt) != INVALID_FILE_ATTRIBUTES ? "present" : "absent");

    printf("== %s (%d failure(s))\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
