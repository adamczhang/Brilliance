/* mp4.cpp -- Mp4Writer (mp4.h) over libavformat, libavcodec and libswscale. */
#include "mm/mp4.h"

#include "rad_internal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef RAD_HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#endif

namespace rad {
namespace mm {

/* Brilliance: mp4.h. Kept until open() puts them on the file. */
void Mp4Writer::set_metadata(const std::string& key, const std::string& value) {
    for (auto& kv : meta_)
        if (kv.first == key) {
            kv.second = value;
            return;
        }
    meta_.emplace_back(key, value);
}

#if defined(_WIN32)
/* Brilliance: mp4.h. The two names are the whole list: AMD's and Microsoft's encoders, whose
 * makers license the codec; the build carries no other H.264 encoder, and none is taken even if
 * a different libavcodec registers one. */
std::vector<std::string> h264_encoder_names(std::string* why) {
    static const char* const kAllowed[] = { "h264_amf", "h264_mf" };
    const char* env = std::getenv("BRILLIANCE_H264_ENCODER");
    if (!env || !*env) return { kAllowed[0], kAllowed[1] };
    std::vector<std::string> out;
    std::string s(env);
    size_t at = 0;
    while (at <= s.size()) {
        size_t end = s.find(',', at);
        if (end == std::string::npos) end = s.size();
        std::string name = s.substr(at, end - at);
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        at = end + 1;
        if (name.empty()) continue;
        bool ok = false;
        for (const char* a : kAllowed) ok = ok || name == a;
        if (!ok) {
            if (why)
                *why = "BRILLIANCE_H264_ENCODER names '" + name + "'; the H.264 encoders this "
                       "build may use are h264_amf and h264_mf";
            return {};
        }
        if (std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
    }
    if (out.empty() && why)
        *why = "BRILLIANCE_H264_ENCODER names no encoder; the H.264 encoders this build may use "
               "are h264_amf and h264_mf";
    return out;
}
#endif

#ifndef RAD_HAVE_FFMPEG

struct Mp4Writer::Impl {};
Mp4Writer::Mp4Writer() = default;
Mp4Writer::~Mp4Writer() = default;
static int no_ffmpeg(std::string* why) {
    if (why) *why = "this build has no libavcodec, so it writes no video";
    return RAD_E_UNSUPPORTED;
}
int Mp4Writer::open(int32_t, int32_t, int64_t, int64_t, int64_t, int64_t, std::string* why) {
    return no_ffmpeg(why);
}
int Mp4Writer::add_frames(const uint8_t*, int32_t, std::string* why) { return no_ffmpeg(why); }
int Mp4Writer::add_audio(const float*, int64_t, std::string* why) { return no_ffmpeg(why); }
int Mp4Writer::finish(std::string*, std::string* why) { return no_ffmpeg(why); }

#else

namespace {

/* The file in memory. The MP4 muxer seeks back to patch sizes and, to put the moov atom first,
 * reopens its output for reading and moves the media data up (FFmpeg's shift_data): the reader is
 * a second handle on the same bytes with a position of its own. */
struct MemFile {
    std::string data;
    int64_t     pos = 0;
};
struct MemReader {
    MemFile* f = nullptr;
    int64_t  pos = 0;
};

/* avio_alloc_context's write_packet takes a const buffer from libavformat 61 and a mutable one
 * before it; the callback is declared as the library at hand wants it. */
#if LIBAVFORMAT_VERSION_MAJOR >= 61
using WriteBuf = const uint8_t*;
#else
using WriteBuf = uint8_t*;
#endif

int mem_write(void* op, WriteBuf buf, int n) {
    MemFile* f = static_cast<MemFile*>(op);
    const int64_t end = f->pos + n;
    if (end > (int64_t)f->data.size()) f->data.resize((size_t)end);
    std::memcpy(&f->data[(size_t)f->pos], buf, (size_t)n);
    f->pos = end;
    return n;
}

int64_t mem_seek(void* op, int64_t off, int whence) {
    MemFile* f = static_cast<MemFile*>(op);
    const int64_t size = (int64_t)f->data.size();
    switch (whence & ~AVSEEK_FORCE) {
    case AVSEEK_SIZE: return size;
    case SEEK_SET: f->pos = off; break;
    case SEEK_CUR: f->pos += off; break;
    case SEEK_END: f->pos = size + off; break;
    default: return AVERROR(EINVAL);
    }
    return f->pos < 0 ? AVERROR(EINVAL) : f->pos;
}

int mem_read(void* op, uint8_t* buf, int n) {
    MemReader* r = static_cast<MemReader*>(op);
    const int64_t size = (int64_t)r->f->data.size();
    if (r->pos >= size) return AVERROR_EOF;
    const int k = (int)std::min<int64_t>(n, size - r->pos);
    std::memcpy(buf, r->f->data.data() + r->pos, (size_t)k);
    r->pos += k;
    return k;
}

int64_t mem_read_seek(void* op, int64_t off, int whence) {
    MemReader* r = static_cast<MemReader*>(op);
    const int64_t size = (int64_t)r->f->data.size();
    switch (whence & ~AVSEEK_FORCE) {
    case AVSEEK_SIZE: return size;
    case SEEK_SET: r->pos = off; break;
    case SEEK_CUR: r->pos += off; break;
    case SEEK_END: r->pos = size + off; break;
    default: return AVERROR(EINVAL);
    }
    return r->pos < 0 ? AVERROR(EINVAL) : r->pos;
}

constexpr int kIoBuf = 1 << 16;

std::string av_why(const char* what, int e) {
    char b[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(e, b, sizeof b);
    return std::string(what) + ": " + b;
}

#if defined(_WIN32)
/* Brilliance: the frame format `c` takes from the writer's swscale: yuv420p where the encoder
 * lists it (both h264_amf and h264_mf do in FFmpeg 8.1), else nv12, else none. */
AVPixelFormat pick_pix_fmt(const AVCodec* c) {
    const AVPixelFormat* fmts = nullptr;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
    const void* cfg = nullptr;
    int n = 0;
    if (avcodec_get_supported_config(nullptr, c, AV_CODEC_CONFIG_PIX_FORMAT, 0, &cfg, &n) >= 0)
        fmts = static_cast<const AVPixelFormat*>(cfg);
#else
    fmts = c->pix_fmts;
#endif
    if (!fmts) return AV_PIX_FMT_YUV420P;   /* the encoder lists none: it takes any */
    bool nv12 = false;
    for (const AVPixelFormat* f = fmts; *f != AV_PIX_FMT_NONE; ++f) {
        if (*f == AV_PIX_FMT_YUV420P) return AV_PIX_FMT_YUV420P;
        nv12 = nv12 || *f == AV_PIX_FMT_NV12;
    }
    return nv12 ? AV_PIX_FMT_NV12 : AV_PIX_FMT_NONE;
}
#endif

}  /* namespace */

struct Mp4Writer::Impl {
    MemFile          file;
    AVFormatContext* oc = nullptr;
    AVCodecContext*  vc = nullptr;
    AVCodecContext*  ac = nullptr;
    AVStream*        vs = nullptr;
    AVStream*        as = nullptr;
    SwsContext*      sws = nullptr;
    AVFrame*         vf = nullptr;
    AVFrame*         af = nullptr;
    AVPacket*        pkt = nullptr;
    int32_t          w = 0, h = 0;
    AVPixelFormat    pix = AV_PIX_FMT_YUV420P;   /* the frames' format, the encoder's (Brilliance) */
    int64_t          channels = 0;
    int64_t          next_frame = 0;
    std::vector<float> audio;   /* interleaved, encoded at finish */
    bool             header = false;

    ~Impl() {
        if (oc) {
            if (oc->pb) {
                av_freep(&oc->pb->buffer);
                avio_context_free(&oc->pb);
            }
            avformat_free_context(oc);
        }
        avcodec_free_context(&vc);
        avcodec_free_context(&ac);
        sws_freeContext(sws);
        av_frame_free(&vf);
        av_frame_free(&af);
        av_packet_free(&pkt);
    }

    /* Every packet `c` has ready, onto stream `st`. */
    int drain(AVCodecContext* c, AVStream* st, std::string* why) {
        for (;;) {
            const int e = avcodec_receive_packet(c, pkt);
            if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) return RAD_OK;
            if (e < 0) {
                if (why) *why = av_why("encoding", e);
                return RAD_E_INVAL;
            }
            /* Brilliance: h264_mf and h264_amf hand out video packets with no duration. A packet
             * reaching the mov muxer without one makes it end the edit list at the last frame's
             * start, which players then drop (win/ffmpeg/check-ffmpeg.c). libavformat fills it in
             * from the stream's avg_frame_rate, set below, so this is the guard should that
             * change: one frame in the codec's time base (1/fps). */
            if (pkt->duration <= 0 && c->codec_type == AVMEDIA_TYPE_VIDEO) pkt->duration = 1;
            av_packet_rescale_ts(pkt, c->time_base, st->time_base);
            pkt->stream_index = st->index;
            const int w2 = av_interleaved_write_frame(oc, pkt);
            if (w2 < 0) {
                if (why) *why = av_why("writing the file", w2);
                return RAD_E_INVAL;
            }
        }
    }
};

/* The reader shift_data opens on the output's own URL; any other URL is refused. */
static int open_reader(AVFormatContext* s, AVIOContext** pb, const char*, int flags, AVDictionary**) {
    if (!(flags & AVIO_FLAG_READ) || (flags & AVIO_FLAG_WRITE)) return AVERROR(EINVAL);
    MemFile* f = static_cast<MemFile*>(s->opaque);
    MemReader* r = new MemReader{ f, 0 };
    uint8_t* buf = static_cast<uint8_t*>(av_malloc(kIoBuf));
    *pb = buf ? avio_alloc_context(buf, kIoBuf, 0, r, mem_read, nullptr, mem_read_seek) : nullptr;
    if (!*pb) {
        av_free(buf);
        delete r;
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int close_reader(AVFormatContext*, AVIOContext* pb) {
    if (!pb) return 0;
    delete static_cast<MemReader*>(pb->opaque);
    av_freep(&pb->buffer);
    avio_context_free(&pb);
    return 0;
}

Mp4Writer::Mp4Writer() : p_(std::make_unique<Impl>()) {}
Mp4Writer::~Mp4Writer() = default;

int Mp4Writer::open(int32_t width, int32_t height, int64_t fps_num, int64_t fps_den,
                    int64_t sample_rate, int64_t channels, std::string* why) {
    Impl& p = *p_;
    /* The encoders print their settings and a file's statistics at FFmpeg's info level, on stderr,
     * for every video; a failure comes back through the calls here, so only errors are let
     * through. */
    static std::once_flag quiet;
    std::call_once(quiet, [] { if (av_log_get_level() > AV_LOG_ERROR) av_log_set_level(AV_LOG_ERROR); });
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1) || fps_num <= 0 || fps_den <= 0) {
        if (why) *why = "the frames must be a positive even size at a positive rate";
        return RAD_E_INVAL;
    }
    p.w = width;
    p.h = height;
    p.channels = channels;
    if (avformat_alloc_output_context2(&p.oc, nullptr, "mp4", nullptr) < 0 || !p.oc) {
        if (why) *why = "this libavformat has no MP4 muxer";
        return RAD_E_UNSUPPORTED;
    }
    p.oc->url = av_strdup("radiance.mp4");
    p.oc->opaque = &p.file;
    p.oc->io_open = open_reader;
    p.oc->io_close2 = close_reader;
    uint8_t* buf = static_cast<uint8_t*>(av_malloc(kIoBuf));
    p.oc->pb = buf ? avio_alloc_context(buf, kIoBuf, 1, &p.file, nullptr, mem_write, mem_seek)
                   : nullptr;
    p.pkt = av_packet_alloc();
    if (!p.oc->url || !p.oc->pb || !p.pkt) {
        if (!p.oc->pb) av_free(buf);
        if (why) *why = "out of memory";
        return RAD_E_NOMEM;
    }

    /* ---- video */
    /* The encoder's settings, whichever it is. */
    auto setup = [&](AVCodecContext* c, const AVCodec* codec, AVPixelFormat pix) {
        c->width = width;
        c->height = height;
        c->pix_fmt = pix;
        c->time_base = AVRational{ (int)fps_den, (int)fps_num };
        c->framerate = AVRational{ (int)fps_num, (int)fps_den };
        c->gop_size = (int)std::max<int64_t>(1, 2 * fps_num / fps_den);
        c->colorspace = AVCOL_SPC_BT709;
        c->color_primaries = AVCOL_PRI_BT709;
        c->color_trc = AVCOL_TRC_BT709;
        c->color_range = AVCOL_RANGE_MPEG;
        if (!std::strcmp(codec->name, "libx264")) {
            /* Constant quality, visually lossless for generated footage. */
            av_opt_set(c->priv_data, "preset", "medium", 0);
            av_opt_set(c->priv_data, "crf", "18", 0);
        } else {
            /* An encoder without a quality mode: 0.2 bits a pixel, about 5 Mbit/s at 1344x768 x 24. */
            c->bit_rate = (int64_t)((double)width * height * fps_num / fps_den * 0.2);
        }
        if (p.oc->oformat->flags & AVFMT_GLOBALHEADER) c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    };
    int e = 0;
#if defined(_WIN32)
    /* Brilliance: h264_amf, then h264_mf, by name (mp4.h); one that is missing or will not open
     * gives way to the next, and the reasons are kept for the refusal when none opens. */
    {
        std::string list_why;
        const std::vector<std::string> names = h264_encoder_names(&list_why);
        if (names.empty()) {
            if (why) *why = list_why;
            return RAD_E_UNSUPPORTED;
        }
        std::string tried;
        for (const std::string& name : names) {
            if (!tried.empty()) tried += "; ";
            const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
            if (!codec) {
                tried += name + ": not in this libavcodec";
                continue;
            }
            const AVPixelFormat pix = pick_pix_fmt(codec);
            if (pix == AV_PIX_FMT_NONE) {
                tried += name + ": takes neither yuv420p nor nv12 frames";
                continue;
            }
            AVCodecContext* c = avcodec_alloc_context3(codec);
            if (!c) {
                if (why) *why = "out of memory";
                return RAD_E_NOMEM;
            }
            setup(c, codec, pix);
            e = avcodec_open2(c, codec, nullptr);
            if (e < 0) {
                tried += av_why(name.c_str(), e);
                avcodec_free_context(&c);
                continue;
            }
            p.vc = c;
            p.pix = pix;
            venc_name_ = name;
            break;
        }
        if (!p.vc) {
            if (why)
                *why = "no H.264 encoder would open (" + tried + "). Video is written only "
                       "through h264_amf, which needs the AMD driver's AMF runtime (amfrt64.dll), "
                       "or h264_mf, which needs Windows Media Foundation's H.264 encoder (on an "
                       "\"N\" edition of Windows, the Media Feature Pack)";
            return RAD_E_UNSUPPORTED;
        }
        RAD_DEBUG("video: H.264 through %s (%s frames)", venc_name_.c_str(),
                  av_get_pix_fmt_name(p.pix));
    }
    p.vs = avformat_new_stream(p.oc, nullptr);
    if (!p.vs) {
        if (why) *why = "out of memory";
        return RAD_E_NOMEM;
    }
#else
    const AVCodec* vcodec = nullptr;
    for (const char* name : { "libx264", "libopenh264" })
        if ((vcodec = avcodec_find_encoder_by_name(name))) break;
    if (!vcodec) vcodec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!vcodec) {
        if (why) *why = "this libavcodec has no H.264 encoder";
        return RAD_E_UNSUPPORTED;
    }
    venc_name_ = vcodec->name;
    p.vs = avformat_new_stream(p.oc, nullptr);
    p.vc = avcodec_alloc_context3(vcodec);
    if (!p.vs || !p.vc) {
        if (why) *why = "out of memory";
        return RAD_E_NOMEM;
    }
    setup(p.vc, vcodec, AV_PIX_FMT_YUV420P);
    e = avcodec_open2(p.vc, vcodec, nullptr);
    if (e < 0) {
        if (why) *why = av_why(("opening the " + venc_name_ + " encoder").c_str(), e);
        return RAD_E_INVAL;
    }
#endif
    avcodec_parameters_from_context(p.vs->codecpar, p.vc);
    p.vs->time_base = p.vc->time_base;
    p.vs->avg_frame_rate = p.vc->framerate;
    p.vf = av_frame_alloc();
    if (!p.vf) return RAD_E_NOMEM;
    p.vf->format = p.pix;
    p.vf->width = width;
    p.vf->height = height;
    if (av_frame_get_buffer(p.vf, 0) < 0) return RAD_E_NOMEM;
    p.sws = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height, p.pix,
                           SWS_BICUBIC | SWS_ACCURATE_RND, nullptr, nullptr, nullptr);
    if (!p.sws) {
        if (why) *why = std::string("libswscale cannot convert RGB to ") + av_get_pix_fmt_name(p.pix);
        return RAD_E_UNSUPPORTED;
    }
    /* Full-range RGB to limited-range BT.709, what the stream's colour tags say. */
    sws_setColorspaceDetails(p.sws, sws_getCoefficients(SWS_CS_DEFAULT), 1,
                             sws_getCoefficients(SWS_CS_ITU709), 0, 0, 1 << 16, 1 << 16);

    /* ---- audio */
    if (channels > 0) {
        const AVCodec* acodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
        if (!acodec) {
            if (why) *why = "this libavcodec has no AAC encoder";
            return RAD_E_UNSUPPORTED;
        }
        p.as = avformat_new_stream(p.oc, nullptr);
        p.ac = avcodec_alloc_context3(acodec);
        if (!p.as || !p.ac) return RAD_E_NOMEM;
        p.ac->sample_fmt = AV_SAMPLE_FMT_FLTP;
        p.ac->sample_rate = (int)sample_rate;
        av_channel_layout_default(&p.ac->ch_layout, (int)channels);
        p.ac->bit_rate = 96000 * channels;
        p.ac->time_base = AVRational{ 1, (int)sample_rate };
        if (p.oc->oformat->flags & AVFMT_GLOBALHEADER) p.ac->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        e = avcodec_open2(p.ac, acodec, nullptr);
        if (e < 0) {
            if (why) *why = av_why("opening the AAC encoder", e);
            return RAD_E_INVAL;
        }
        avcodec_parameters_from_context(p.as->codecpar, p.ac);
        p.as->time_base = p.ac->time_base;
        p.af = av_frame_alloc();
        if (!p.af) return RAD_E_NOMEM;
        p.af->format = AV_SAMPLE_FMT_FLTP;
        p.af->sample_rate = p.ac->sample_rate;
        av_channel_layout_copy(&p.af->ch_layout, &p.ac->ch_layout);
        p.af->nb_samples = p.ac->frame_size > 0 ? p.ac->frame_size : 1024;
        if (av_frame_get_buffer(p.af, 0) < 0) return RAD_E_NOMEM;
    }

    /* Brilliance: the file's metadata (set_metadata), which the muxer writes as the standard
     * MP4 (iTunes-style) atoms it maps those keys to. */
    for (const auto& kv : meta_)
        if (av_dict_set(&p.oc->metadata, kv.first.c_str(), kv.second.c_str(), 0) < 0) {
            if (why) *why = "out of memory";
            return RAD_E_NOMEM;
        }

    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "movflags", "+faststart", 0);
    e = avformat_write_header(p.oc, &opts);
    av_dict_free(&opts);
    if (e < 0) {
        if (why) *why = av_why("writing the MP4 header", e);
        return RAD_E_INVAL;
    }
    p.header = true;
    return RAD_OK;
}

int Mp4Writer::add_frames(const uint8_t* rgb, int32_t n, std::string* why) {
    Impl& p = *p_;
    if (!p.header) {
        if (why) *why = "the writer is not open";
        return RAD_E_STATE;
    }
    const int64_t frame_bytes = (int64_t)p.w * p.h * 3;
    for (int32_t i = 0; i < n; ++i) {
        if (av_frame_make_writable(p.vf) < 0) return RAD_E_NOMEM;
        const uint8_t* src[1] = { rgb + i * frame_bytes };
        const int stride[1] = { p.w * 3 };
        sws_scale(p.sws, src, stride, 0, p.h, p.vf->data, p.vf->linesize);
        p.vf->pts = p.next_frame++;
        const int e = avcodec_send_frame(p.vc, p.vf);
        if (e < 0) {
            if (why) *why = av_why("encoding a frame", e);
            return RAD_E_INVAL;
        }
        RAD_TRY(p.drain(p.vc, p.vs, why));
    }
    return RAD_OK;
}

int Mp4Writer::add_audio(const float* samples, int64_t n_frames, std::string* why) {
    Impl& p = *p_;
    if (!p.header || !p.ac) {
        if (why) *why = "the writer has no soundtrack";
        return RAD_E_STATE;
    }
    p.audio.insert(p.audio.end(), samples, samples + n_frames * p.channels);
    return RAD_OK;
}

int Mp4Writer::finish(std::string* out, std::string* why) {
    Impl& p = *p_;
    if (!p.header) {
        if (why) *why = "the writer is not open";
        return RAD_E_STATE;
    }
    if (p.ac) {
        /* The soundtrack in the encoder's frames, deinterleaved; the last frame padded with
         * silence, which the encoder's own length bookkeeping trims. */
        const int64_t total = (int64_t)p.audio.size() / p.channels;
        const int fs = p.af->nb_samples;
        for (int64_t at = 0; at < total; at += fs) {
            if (av_frame_make_writable(p.af) < 0) return RAD_E_NOMEM;
            const int64_t k = std::min<int64_t>(fs, total - at);
            p.af->nb_samples = (int)k;
            for (int64_t c = 0; c < p.channels; ++c) {
                float* d = reinterpret_cast<float*>(p.af->data[c]);
                for (int64_t i = 0; i < k; ++i)
                    d[i] = p.audio[(size_t)((at + i) * p.channels + c)];
            }
            p.af->pts = at;
            const int e = avcodec_send_frame(p.ac, p.af);
            if (e < 0) {
                if (why) *why = av_why("encoding the soundtrack", e);
                return RAD_E_INVAL;
            }
            RAD_TRY(p.drain(p.ac, p.as, why));
            p.af->nb_samples = fs;
        }
        avcodec_send_frame(p.ac, nullptr);
        RAD_TRY(p.drain(p.ac, p.as, why));
    }
    avcodec_send_frame(p.vc, nullptr);
    RAD_TRY(p.drain(p.vc, p.vs, why));
    const int e = av_write_trailer(p.oc);
    if (e < 0) {
        if (why) *why = av_why("finishing the MP4 file", e);
        return RAD_E_INVAL;
    }
    avio_flush(p.oc->pb);
    *out = std::move(p.file.data);
    p.file.data.clear();
    p.header = false;
    return RAD_OK;
}

#endif  /* RAD_HAVE_FFMPEG */

}  /* namespace mm */
}  /* namespace rad */
