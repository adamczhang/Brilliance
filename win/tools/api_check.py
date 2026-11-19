#!/usr/bin/env python3
"""Brilliance: checks of the APIs beside chat -- embeddings, rerank, speech, transcription, images --
against a live server.

qualify_api.py covers a chat model. This covers the models that serve one endpoint each. Every check
sends real requests and judges the answer; the run prints one line a check and exits 0 only when
every check that ran passed. Standard library only, so the toolchain's CPython runs it:

  .toolchain\\python\\python.exe -I win\\tools\\api_check.py --url http://127.0.0.1:8120 \\
      --out out\\apis\\embgemma2-1.jsonl --save out\\apis\\out

Without --only, the checks are chosen from the model's capabilities at /v1/models:

  embeddings          related sentences score above unrelated ones by cosine similarity; one vector
                      an input, unit length, finite; a batch's rows equal the same texts sent alone;
                      a repeat is identical; `dimensions` is the renormalised prefix; base64 holds
                      the same floats; an input past the step is refused, not truncated; requests
                      in flight together each get their own text's vector
  embeddings_image    (capability image) pictures made here: the same picture twice is one vector, a
                      red picture is nearer "red" than "blue" and a blue one the reverse, and text
                      and a picture together make one vector
  embeddings_audio    (capability audio) clips made here: a tone is nearer the same tone played
                      quieter than it is to noise; the content-part and the chat form agree; with
                      --audio FILE --audio-text T --audio-other U, the recording is nearer T than U
  embeddings_video    (capability video, --video FILE) a clip makes one finite unit vector
  embeddings_compare  (--dump FILE, --compare FILE) the vectors of a fixed set of texts, pictures
                      and clips, written to a file and held against an earlier one: the same
                      container on a card and on the CPU kernels gives the same vectors
  rerank              the relevant document ranks first; scores finite and descending; top_n and
                      return_documents:false are honoured
  rerank_batch        a repeat is identical; each document scores as it does alone, reversed, as
                      content parts and among 32 documents (upstream's 1e-5 gate); usage counts
                      every pair, with top_n too; requests in flight together get their own scores
  rerank_limits       a pair past the context is a 400, not truncated; no documents, no query or a
                      number as a document is a 400; top_n 0 returns all
  rerank_image        (capability image) a red picture outranks a blue one for "red", and the reverse
  speech              by default on Windows: the build refuses speech (Brilliance 0.3.0, the owner's
                      decision): /v1/audio/speech and /audio/speech answer 501 with an error body,
                      type not_implemented_error, saying "not supported in this build", and the
                      /v1/audio/speech/stream socket sends that as an error event and closes 1008.
                      With --speech served (the default elsewhere): /v1/audio/speech returns a WAV
                      of plausible length for its text that is not silence; saved under --save as
                      speech.wav
  transcription       /v1/audio/transcriptions over --audio FILE: most of --audio-text's words come
                      back. Run it on speech.wav from a speech server for the round trip
  transcription_forms json, text, verbose_json and the stream give one transcript (the stream's
                      deltas join to it; verbose_json's duration is the WAV's); a repeat and 8 at a
                      time are identical; srt, a temperature, an unknown language and a file that
                      is not audio are 400s
  transcription_alt   (--audio-alt FILE ...) the same recording as mp3, flac, ogg, m4a, ...: most of
                      --audio-text's words from each
  realtime            (--audio a 16-bit mono WAV) /v1/realtime over a WebSocket, OpenAI's live
                      transcription: streamed at real time, words settle before the commit, the
                      deltas start the turn's transcript, which equals the upload route's; the same
                      audio as binary frames gives it again; a bad rate is an error event; clear
                      and close are answered
  images              /v1/images/generations returns a PNG that decodes, of the size asked, that is
                      not one flat colour, and the same seed gives the same picture
  images_forms        JPEG and WebP of the size asked; b64_json PNG by default; response_format url
                      either served or a 400 naming it
  images_sizes        --image-size2 (non-square) comes back as asked, 1000x700 rounded down to the
                      model's grid, the response's size the picture's
  images_n            n 2: two pictures, seeds s and s + 1, each the bytes of its seed alone; two
                      sizes in flight together each give their solo bytes
  images_limits       malformed sizes, prompts, n and formats are 400s with an error body
  refusals            every endpoint here the model does not serve answers 501 with an error body
  selftest            no server: the PNG and WAV code here reads back what it writes

Run on the test machine so far: the embeddings checks, refusals and selftest, on EmbeddingGemma 2 and a chat
model (2026-11-17); the rerank checks on Nemotron 3.5 Rerank 8B BF16, and transcription and realtime
on Qwen3-ASR 1.7B over speech from Windows' SAPI voices (2026-11-18); the images checks on Stable
Diffusion XL base 1.0 bf16 at 1024x1024 and 832x1216 (2026-11-18). speech has not met a model yet, in
either form.

  api_check.py --url http://127.0.0.1:8130 --transcribes --audio clip.wav --audio-alt clip.mp3       --audio-alt clip.flac --audio-text "what the clip says"
"""
import argparse
import base64
import hashlib
import http.client
import io
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import urllib.parse
import wave
import zlib
from concurrent.futures import ThreadPoolExecutor

ARGS = None
RESULTS = []
LOG = None


# ---------------------------------------------------------------- transport

def request(method, path, body=None, ctype="application/json", timeout=600):
    """One request, whole. Returns (status, content type, bytes)."""
    u = urllib.parse.urlparse(ARGS.url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    h = {}
    if body is not None:
        h["Content-Type"] = ctype
    if ARGS.key:
        h["Authorization"] = "Bearer " + ARGS.key
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, r.getheader("Content-Type") or "", data


def post(path, obj, timeout=600):
    """A JSON request. Returns (status, parsed JSON or None, bytes)."""
    status, _, data = request("POST", path, json.dumps(obj).encode(), timeout=timeout)
    try:
        return status, json.loads(data.decode("utf-8")), data
    except (ValueError, UnicodeDecodeError):
        return status, None, data


def post_form(path, fields, files, timeout=600):
    """multipart/form-data, as OpenAI's SDKs send a transcription. files: name -> (filename, bytes)."""
    body, ctype = form_body(fields, files)
    status, _, data = request("POST", path, body, ctype, timeout)
    try:
        return status, json.loads(data.decode("utf-8")), data
    except (ValueError, UnicodeDecodeError):
        return status, None, data


def form_body(fields, files):
    """(body, content type) of a multipart/form-data request."""
    boundary = "----brl" + os.urandom(8).hex()
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                   % (boundary, k, v)).encode())
    for k, (name, data) in files.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                   "Content-Type: application/octet-stream\r\n\r\n" % (boundary, k, name)).encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(("--%s--\r\n" % boundary).encode())
    return out.getvalue(), "multipart/form-data; boundary=" + boundary


def err_text(j, data):
    if isinstance(j, dict) and isinstance(j.get("error"), dict):
        return str(j["error"].get("message"))
    return data[:200].decode("utf-8", "replace")


# ---------------------------------------------------------------- results

class Fail(Exception):
    pass


def need(cond, what):
    if not cond:
        raise Fail(what)


def record(name, ok, detail, facts=None):
    RESULTS.append((name, ok))
    print("%s  %-18s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if LOG:
        LOG.write(json.dumps({"check": name, "pass": ok, "detail": detail, "facts": facts or {},
                              "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        LOG.flush()


def run(name, fn):
    facts = {}
    try:
        detail = fn(facts)
        record(name, True, detail, facts)
    except Fail as e:
        record(name, False, str(e), facts)
    except Exception as e:  # a transport error is a failed check, not a crashed run
        record(name, False, "%s: %s" % (type(e).__name__, e), facts)


# ---------------------------------------------------------------- media made here

def png_bytes(w, h, rgb_at):
    """An 8-bit RGB PNG; rgb_at(x, y) gives a pixel."""
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw.extend(rgb_at(x, y))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def png_decode(data):
    """Decode a non-interlaced 8- or 16-bit PNG. Returns (w, h, channels, pixel bytes). Every chunk's
    CRC is checked and every row unfiltered, so a truncated or damaged file fails here."""
    need(data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG (starts %r)" % data[:8])
    pos, idat, head, ended = 8, [], None, False
    while pos + 8 <= len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        need(len(body) == n and pos + 12 + n <= len(data), "PNG chunk %r is cut short" % kind)
        crc, = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        need(crc == zlib.crc32(kind + body) & 0xFFFFFFFF, "PNG chunk %r fails its CRC" % kind)
        if kind == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            ended = True
            break
        pos += 12 + n
    need(head is not None and ended, "PNG without IHDR or IEND")
    w, h, depth, ctype, _, _, interlace = head
    need(interlace == 0 and depth in (8, 16), "PNG form not read here (depth %d, interlace %d)"
         % (depth, interlace))
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    need(channels is not None, "PNG colour type %d not read here" % ctype)
    bpp = channels * depth // 8
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    need(len(raw) == h * (stride + 1), "PNG holds %d bytes of rows, %dx%d needs %d"
         % (len(raw), w, h, h * (stride + 1)))
    out = bytearray(h * stride)
    prev = bytearray(stride)
    for y in range(h):
        f = raw[y * (stride + 1)]
        row = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        need(f <= 4, "PNG row %d has filter %d" % (y, f))
        if f == 1:
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 255
        elif f == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((a + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out[y * stride:(y + 1) * stride] = row
        prev = row
    return w, h, channels, bytes(out)


def wav_bytes(samples, rate=16000):
    """16-bit mono WAV from floats in [-1, 1]."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % len(samples),
                                  *[max(-32767, min(32767, int(s * 32767))) for s in samples]))
    return buf.getvalue()


def tone(hz, seconds, amp, rate=16000):
    return [amp * math.sin(2 * math.pi * hz * i / rate) for i in range(int(seconds * rate))]


def noise(seconds, amp, rate=16000, seed=12345):
    out, x = [], seed
    for _ in range(int(seconds * rate)):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(amp * (x / 0x3FFFFFFF - 1.0))
    return out


def wav_stats(data):
    """(seconds, rate, channels, RMS in 0..1, peak in 0..1) of a 16-bit PCM WAV. A streamed WAV's
    header may carry a placeholder length, so the samples are read from the data chunk to the end."""
    need(data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a WAV (starts %r)" % data[:12])
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        kind, n = struct.unpack("<4sI", data[pos:pos + 8])
        if kind == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[pos + 8:pos + 24])
        elif kind == b"data":
            pcm = data[pos + 8:] if n in (0, 0xFFFFFFFF) or pos + 8 + n > len(data) \
                else data[pos + 8:pos + 8 + n]
            break
        pos += 8 + n + (n & 1)
    need(fmt is not None and pcm is not None, "WAV without a fmt or a data chunk")
    code, channels, rate, _, _, bits = fmt
    need(code == 1 and bits == 16, "WAV form not read here (format %d, %d bits)" % (code, bits))
    count = len(pcm) // 2
    need(count > 0, "WAV holds no samples")
    vals = struct.unpack("<%dh" % count, pcm[:count * 2])
    rms = math.sqrt(sum(v * v for v in vals) / count) / 32768.0
    peak = max(abs(v) for v in vals) / 32768.0
    return count / channels / rate, rate, channels, rms, peak


def data_url(mime, data):
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))


# ---------------------------------------------------------------- vectors

def cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def norm(a):
    return math.sqrt(sum(x * x for x in a))


def embed(body, expect=None):
    """POST /v1/embeddings; returns the vectors in input order, checked finite and unit length."""
    b = dict(body)
    if ARGS.model:
        b["model"] = ARGS.model
    status, j, data = post("/v1/embeddings", b)
    need(status == 200 and j, "/v1/embeddings: %d %s" % (status, err_text(j, data)))
    rows = j.get("data")
    need(isinstance(rows, list) and rows, "/v1/embeddings: no data")
    if expect is not None:
        need(len(rows) == expect, "/v1/embeddings: %d vectors for %d inputs" % (len(rows), expect))
    need([r.get("index") for r in rows] == list(range(len(rows))), "/v1/embeddings: index order")
    vecs = []
    for r in rows:
        v = r.get("embedding")
        if isinstance(v, str):
            raw = base64.b64decode(v)
            v = list(struct.unpack("<%df" % (len(raw) // 4), raw))
        need(isinstance(v, list) and v, "/v1/embeddings: an embedding is missing")
        need(all(isinstance(x, (int, float)) and math.isfinite(x) for x in v),
             "/v1/embeddings: a non-finite value")
        need(abs(norm(v) - 1.0) < 1e-3, "/v1/embeddings: a vector of length %.5f, not 1" % norm(v))
        vecs.append(v)
    need(len(set(len(v) for v in vecs)) == 1, "/v1/embeddings: vectors of different sizes")
    return vecs, j


TRIPLETS = [
    ("How do I bake sourdough bread at home?",
     "Mix flour, water, salt and a ripe starter, let the dough rise overnight, then bake it in a "
     "hot Dutch oven.",
     "The central bank raised its interest rate by a quarter point to slow inflation."),
    ("What causes the seasons on Earth?",
     "The tilt of the planet's axis means each hemisphere gets more direct sunlight for half of "
     "the year.",
     "A goalkeeper may handle the ball only inside the penalty area."),
    ("symptoms of a failing car battery",
     "The engine cranks slowly, the headlights dim at idle and the dashboard warning light comes "
     "on.",
     "Whales migrate thousands of kilometres between feeding and breeding waters."),
    ("Wie funktioniert ein Elektromotor?",
     "Strom in einer Spule erzeugt ein Magnetfeld, das den Rotor in Drehung versetzt.",
     "Der Roman erschien 1925 und spielt im New York der zwanziger Jahre."),
]


def with_prompt(body, name):
    if name:
        body = dict(body)
        body["prompt_name"] = name
    return body


def check_embeddings(facts):
    queries = [t[0] for t in TRIPLETS]
    docs = [t[1] for t in TRIPLETS] + [t[2] for t in TRIPLETS]
    qv, j = embed(with_prompt({"input": queries}, ARGS.query_prompt), len(queries))
    dv, _ = embed(with_prompt({"input": docs}, ARGS.doc_prompt), len(docs))
    dim = len(qv[0])
    facts["dim"] = dim
    usage = j.get("usage") or {}
    need(usage.get("prompt_tokens", 0) > 0, "no usage.prompt_tokens")
    margins = []
    for i in range(len(TRIPLETS)):
        rel, unrel = cos(qv[i], dv[i]), cos(qv[i], dv[len(TRIPLETS) + i])
        margins.append((round(rel, 4), round(unrel, 4)))
        need(rel > unrel + 0.05, "triplet %d: related %.4f is not above unrelated %.4f"
             % (i, rel, unrel))
        # Retrieval: among all eight documents the query's own comes first.
        best = max(range(len(dv)), key=lambda k: cos(qv[i], dv[k]))
        need(best == i, "triplet %d: document %d ranks first, not its own" % (i, best))
    facts["related_unrelated"] = margins

    # A batch's row is the vector the same text gets alone, and a repeat is the same vector.
    alone, _ = embed(with_prompt({"input": queries[1]}, ARGS.query_prompt), 1)
    c = cos(alone[0], qv[1])
    facts["batch_vs_alone"] = c
    need(c > 0.9995, "a batch's row and the same text alone differ (cosine %.6f)" % c)
    again, _ = embed(with_prompt({"input": queries[1]}, ARGS.query_prompt), 1)
    need(again[0] == alone[0], "the same request twice gave two vectors (cosine %.6f)"
         % cos(again[0], alone[0]))

    # base64 carries the same floats.
    b64, _ = embed(with_prompt({"input": queries[1], "encoding_format": "base64"},
                               ARGS.query_prompt), 1)
    worst = max(abs(x - y) for x, y in zip(b64[0], alone[0]))
    need(len(b64[0]) == dim and worst < 1e-6, "base64 differs from the float form by %g" % worst)

    # Matryoshka: the first N dimensions, renormalised -- or a refusal that says so.
    small = dim // 3 if dim % 3 == 0 else dim // 2
    b = with_prompt({"input": queries[1], "dimensions": small}, ARGS.query_prompt)
    if ARGS.model:
        b["model"] = ARGS.model
    status, jd, data = post("/v1/embeddings", b)
    if status == 200:
        v = jd["data"][0]["embedding"]
        need(len(v) == small, "dimensions %d gave %d values" % (small, len(v)))
        c = cos(v, alone[0][:small])
        need(abs(norm(v) - 1.0) < 1e-3 and c > 0.9999,
             "dimensions %d is not the renormalised prefix (cosine %.6f)" % (small, c))
        facts["dimensions"] = "%d: the renormalised prefix" % small
    else:
        need(400 <= status < 500, "dimensions: %d %s" % (status, err_text(jd, data)))
        facts["dimensions"] = "refused: " + err_text(jd, data)

    # An input longer than the step is refused, never cut.
    b = {"input": "the quick brown fox jumps over the lazy dog " * 40000}
    if ARGS.model:
        b["model"] = ARGS.model
    status, jl, data = post("/v1/embeddings", b)
    need(400 <= status < 500, "a 1.7 MB input: %d %s" % (status, err_text(jl, data)[:120]))
    facts["overlong"] = "%d %s" % (status, err_text(jl, data)[:160])
    # Requests in flight together share steps: each still gets its own text's vector.
    def one(k):
        if k % 2:
            v, _ = embed(with_prompt({"input": docs[k % len(docs)]}, ARGS.doc_prompt), 1)
            return cos(v[0], dv[k % len(docs)])
        v, _ = embed(with_prompt({"input": queries[k % len(queries)]}, ARGS.query_prompt), 1)
        return cos(v[0], qv[k % len(queries)])
    with ThreadPoolExecutor(ARGS.concurrency) as pool:
        together = list(pool.map(one, range(4 * ARGS.concurrency)))
    facts["concurrent_min_cosine"] = min(together)
    need(min(together) > 0.9995, "%d requests %d at a time: one differs from its text's vector "
         "(cosine %.6f)" % (len(together), ARGS.concurrency, min(together)))
    return ("dim %d; related/unrelated cosines %s; batch = alone (%.6f); repeat identical; base64 "
            "equal; dimensions %s; overlong input %d; %d requests %d at a time each got its own "
            "vector (min cosine %.6f)" % (dim, margins, facts["batch_vs_alone"],
                                          facts["dimensions"], status, len(together),
                                          ARGS.concurrency, min(together)))


def image_part(data, mime="image/png"):
    return {"type": "image_url", "image_url": {"url": data_url(mime, data)}}


def check_embeddings_image(facts):
    red = png_bytes(224, 224, lambda x, y: (220, 20, 20))
    blue = png_bytes(224, 224, lambda x, y: (20, 40, 220))
    rv, _ = embed({"input": [[image_part(red)], [image_part(blue)], [image_part(red)]]}, 3)
    need(cos(rv[0], rv[2]) > 0.9999, "the same picture twice in a batch: cosine %.6f"
         % cos(rv[0], rv[2]))
    facts["red_vs_blue_picture"] = cos(rv[0], rv[1])
    need(cos(rv[0], rv[1]) < 0.999, "a red and a blue picture got one vector")
    tv, _ = embed(with_prompt({"input": ["a solid red picture", "a solid blue picture"]},
                              ARGS.query_prompt), 2)
    rr, rb = cos(rv[0], tv[0]), cos(rv[0], tv[1])
    br, bb = cos(rv[1], tv[0]), cos(rv[1], tv[1])
    facts["cross_modal"] = {"red_picture": [rr, rb], "blue_picture": [br, bb]}
    need(rr > rb, "the red picture is nearer 'blue' (%.4f) than 'red' (%.4f)" % (rb, rr))
    need(bb > br, "the blue picture is nearer 'red' (%.4f) than 'blue' (%.4f)" % (br, bb))
    mixed, _ = embed({"input": [{"type": "text", "text": "a picture of the colour "},
                                image_part(red)]}, 1)
    need(cos(mixed[0], rv[0]) < 0.9999, "text beside a picture did not change the vector")
    return ("red picture: 'red' %.4f > 'blue' %.4f; blue picture: 'blue' %.4f > 'red' %.4f; the "
            "two pictures %.4f apart; text and picture make one vector"
            % (rr, rb, bb, br, facts["red_vs_blue_picture"]))


def audio_part(data, fmt="wav"):
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(data).decode("ascii"), "format": fmt}}


def check_embeddings_audio(facts):
    loud = wav_bytes(tone(440, 2.0, 0.5))
    quiet = wav_bytes(tone(440, 2.0, 0.25))
    hiss = wav_bytes(noise(2.0, 0.5))
    v, _ = embed({"input": [[audio_part(loud)], [audio_part(quiet)], [audio_part(hiss)],
                            [audio_part(loud)]]}, 4)
    need(cos(v[0], v[3]) > 0.9999, "the same clip twice in a batch: cosine %.6f" % cos(v[0], v[3]))
    same, other = cos(v[0], v[1]), cos(v[0], v[2])
    facts["tone_vs_quieter_tone"] = same
    facts["tone_vs_noise"] = other
    need(same > other + 0.02, "a tone is no nearer itself played quieter (%.4f) than noise (%.4f)"
         % (same, other))
    chat, _ = embed({"messages": [{"role": "user", "content": [audio_part(loud)]}]}, 1)
    facts["chat_form_vs_part"] = cos(chat[0], v[0])
    detail = ("tone vs the same tone quieter %.4f > tone vs noise %.4f; chat form vs content part "
              "%.4f" % (same, other, facts["chat_form_vs_part"]))
    if ARGS.audio and ARGS.audio_text and ARGS.audio_other:
        with open(ARGS.audio, "rb") as f:
            clip = f.read()
        fmt = os.path.splitext(ARGS.audio)[1].lstrip(".").lower() or "wav"
        a, _ = embed({"input": [[audio_part(clip, fmt)]]}, 1)
        t, _ = embed(with_prompt({"input": [ARGS.audio_text, ARGS.audio_other]},
                                 ARGS.query_prompt), 2)
        hit, miss = cos(a[0], t[0]), cos(a[0], t[1])
        facts["recording"] = {"file": ARGS.audio, "its_text": hit, "other_text": miss}
        need(hit > miss, "the recording is nearer the other text (%.4f) than its own (%.4f)"
             % (miss, hit))
        detail += "; recording vs its text %.4f > the other text %.4f" % (hit, miss)
    return detail


def check_embeddings_video(facts):
    with open(ARGS.video, "rb") as f:
        clip = f.read()
    part = {"type": "video_url", "video_url": {"url": data_url("video/mp4", clip)}}
    v, j = embed({"input": [[part]]}, 1)
    again, _ = embed({"input": [[part]]}, 1)
    need(cos(v[0], again[0]) > 0.9999, "the same clip twice: cosine %.6f" % cos(v[0], again[0]))
    facts["prompt_tokens"] = (j.get("usage") or {}).get("prompt_tokens")
    return "%s: one vector of %d, %s prompt tokens, repeatable" % (
        os.path.basename(ARGS.video), len(v[0]), facts["prompt_tokens"])


def check_embeddings_compare(facts, caps):
    """The vectors of a fixed set of inputs, written to --dump and held against --compare: the same
    container on another backend (a card and the CPU kernels) gives the same vectors."""
    names = ["query%d" % i for i in range(len(TRIPLETS))]
    vecs, _ = embed(with_prompt({"input": [t[0] for t in TRIPLETS]}, ARGS.query_prompt),
                    len(TRIPLETS))
    names += ["related%d" % i for i in range(len(TRIPLETS))]
    names += ["unrelated%d" % i for i in range(len(TRIPLETS))]
    more, _ = embed(with_prompt({"input": [t[1] for t in TRIPLETS] + [t[2] for t in TRIPLETS]},
                                ARGS.doc_prompt), 2 * len(TRIPLETS))
    vecs += more
    if caps.get("image"):
        more, _ = embed({"input": [[image_part(png_bytes(224, 224, lambda x, y: (220, 20, 20)))],
                                   [image_part(png_bytes(160, 96, lambda x, y: (x, y * 2, 128)))]]},
                        2)
        names += ["image_red", "image_gradient"]
        vecs += more
    if caps.get("audio"):
        more, _ = embed({"input": [[audio_part(wav_bytes(tone(440, 2.0, 0.5)))],
                                   [audio_part(wav_bytes(noise(2.0, 0.5)))]]}, 2)
        names += ["audio_tone", "audio_noise"]
        vecs += more
    if caps.get("video") and ARGS.video:
        with open(ARGS.video, "rb") as f:
            clip = f.read()
        more, _ = embed({"input": [[{"type": "video_url",
                                     "video_url": {"url": data_url("video/mp4", clip)}}]]}, 1)
        names += ["video"]
        vecs += more
    mine = dict(zip(names, vecs))
    if ARGS.dump:
        os.makedirs(os.path.dirname(os.path.abspath(ARGS.dump)), exist_ok=True)
        with open(ARGS.dump, "w", encoding="utf-8") as f:
            json.dump(mine, f)
    detail = "%d vectors" % len(mine) + (" written to " + ARGS.dump if ARGS.dump else "")
    if ARGS.compare:
        with open(ARGS.compare, "r", encoding="utf-8") as f:
            theirs = json.load(f)
        both = [n for n in names if n in theirs]
        need(both, "%s holds none of these inputs" % ARGS.compare)
        cosines = {n: cos(mine[n], theirs[n]) for n in both}
        facts["cosines"] = cosines
        worst = min(cosines, key=cosines.get)
        need(cosines[worst] >= ARGS.compare_min, "%s differs from %s: cosine %.6f (floor %.4f)"
             % (worst, ARGS.compare, cosines[worst], ARGS.compare_min))
        detail += "; %d held against %s: min cosine %.6f (%s), mean %.6f" % (
            len(both), ARGS.compare, cosines[worst], worst, sum(cosines.values()) / len(both))
    return detail


# ---------------------------------------------------------------- rerank

RERANK_QUERY = "Where can I borrow books?"
RERANK_DOCS = [
    "The stadium holds forty thousand people for a football match.",
    "A public library lends books, films and music to anyone with a card.",
    "Bread dough rises faster in a warm kitchen.",
    "The ferry crosses the strait every half hour in summer.",
]


def check_rerank(facts):
    b = {"query": RERANK_QUERY, "documents": RERANK_DOCS}
    if ARGS.model:
        b["model"] = ARGS.model
    status, j, data = post("/v1/rerank", b)
    need(status == 200 and j, "/v1/rerank: %d %s" % (status, err_text(j, data)))
    res = j.get("results")
    need(isinstance(res, list) and len(res) == len(RERANK_DOCS), "/v1/rerank: results missing")
    scores = [r.get("relevance_score") for r in res]
    need(all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores), "a bad score")
    need(scores == sorted(scores, reverse=True), "results are not in descending order")
    need(sorted(r.get("index") for r in res) == list(range(len(RERANK_DOCS))), "indices")
    facts["scores"] = {r["index"]: r["relevance_score"] for r in res}
    need(res[0]["index"] == 1, "document %d ranks first, not the library (scores %s)"
         % (res[0]["index"], facts["scores"]))
    need(res[0].get("document") is not None, "the document is not returned")
    b2 = dict(b, top_n=2, return_documents=False)
    status, j2, data = post("/rerank", b2)
    need(status == 200 and j2, "/rerank with top_n: %d %s" % (status, err_text(j2, data)))
    res2 = j2.get("results") or []
    need(len(res2) == 2 and res2[0]["index"] == 1 and "document" not in res2[0],
         "top_n 2 / return_documents false not honoured: %s" % res2)
    return "the library document first (%.3f), next %.3f; top_n and return_documents honoured" % (
        scores[0], scores[1])


# Upstream's individual/batched gate for the reranker (docs/ARCHITECTURES.md, nemotron35_embed).
RERANK_TOL = 1e-5


def rerank(body, path="/v1/rerank"):
    """One rerank request, its answer checked whole. Returns ({index: score}, the JSON)."""
    b = dict(body)
    if ARGS.model:
        b["model"] = ARGS.model
    status, j, data = post(path, b)
    need(status == 200 and j, "%s: %d %s" % (path, status, err_text(j, data)))
    res = j.get("results")
    n = len(b["documents"])
    want = min(n, b["top_n"]) if b.get("top_n") else n
    need(isinstance(res, list) and len(res) == want,
         "%s: %s results for %d documents" % (path, len(res) if isinstance(res, list) else
                                              "no", n))
    scores = [r.get("relevance_score") for r in res]
    need(all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores), "a bad score")
    need(scores == sorted(scores, reverse=True), "results are not in descending order")
    idx = [r.get("index") for r in res]
    need(len(set(idx)) == len(idx) and all(isinstance(i, int) and 0 <= i < n for i in idx),
         "indices %s for %d documents" % (idx, n))
    return {r["index"]: r["relevance_score"] for r in res}, j


def filler_docs(n):
    """n distinct sentences about nothing a library query should match."""
    things = ["harbour", "orchard", "glacier", "bakery", "railway", "vineyard", "lighthouse"]
    acts = ["was repainted", "flooded", "closed early", "hired two people"]
    return ["On day %d the %s %s." % (k + 1, things[k % len(things)], acts[k // len(things) % 4])
            for k in range(n)]


def check_rerank_batch(facts):
    """A batch scores each document as it is scored alone, in any order and in any company; a repeat
    is identical; requests in flight together each get their own documents' scores."""
    q = RERANK_QUERY
    base, j = rerank({"query": q, "documents": RERANK_DOCS})
    n = len(RERANK_DOCS)
    again, _ = rerank({"query": q, "documents": RERANK_DOCS})
    need(again == base, "a repeat changed the scores: %s, then %s" % (base, again))
    worst = {}
    tokens = 0
    for i, d in enumerate(RERANK_DOCS):
        s, jo = rerank({"query": q, "documents": [d]})
        worst["alone"] = max(worst.get("alone", 0.0), abs(s[0] - base[i]))
        tokens += (jo.get("usage") or {}).get("prompt_tokens") or 0
    used = (j.get("usage") or {}).get("prompt_tokens")
    facts["usage"] = {"batch": used, "sum_alone": tokens}
    need(used == tokens, "the batch's usage is %s tokens; its pairs alone are %d" % (used, tokens))
    top1, j1 = rerank({"query": q, "documents": RERANK_DOCS, "top_n": 1})
    need((j1.get("usage") or {}).get("prompt_tokens") == used,
         "top_n 1 reports %s tokens, the whole batch %s: usage counts every pair scored"
         % ((j1.get("usage") or {}).get("prompt_tokens"), used))
    need(list(top1) == [max(base, key=base.get)], "top_n 1 is %s" % top1)
    rev, _ = rerank({"query": q, "documents": RERANK_DOCS[::-1]})
    worst["reversed"] = max(abs(rev[k] - base[n - 1 - k]) for k in range(n))
    parts, jp = rerank({"query": q, "documents": [{"content": [{"type": "text", "text": d}]}
                                                  for d in RERANK_DOCS]})
    worst["content_parts"] = max(abs(parts[i] - base[i]) for i in range(n))
    big_docs = filler_docs(14) + RERANK_DOCS + filler_docs(28)[14:]
    big, _ = rerank({"query": q, "documents": big_docs})
    worst["among_32"] = max(abs(big[14 + i] - base[i]) for i in range(n))
    need(max(big, key=big.get) == 14 + 1, "among 32 documents, %d ranks first, not the library"
         % max(big, key=big.get))
    facts["worst_difference"] = worst
    for k, v in worst.items():
        need(v <= RERANK_TOL, "%s: a score differs from the four-document batch's by %.3g "
             "(the gate is %g)" % (k, v, RERANK_TOL))

    def one(k):
        rot = RERANK_DOCS[k % n:] + RERANK_DOCS[:k % n]
        s, _ = rerank({"query": q, "documents": rot})
        return max(abs(s[i] - base[(i + k) % n]) for i in range(n))
    with ThreadPoolExecutor(ARGS.concurrency) as pool:
        together = list(pool.map(one, range(4 * ARGS.concurrency)))
    facts["concurrent_worst"] = max(together)
    need(max(together) <= RERANK_TOL, "%d requests %d at a time: a score differs by %.3g"
         % (len(together), ARGS.concurrency, max(together)))
    return ("repeat identical; alone, reversed, as content parts and among 32 documents within "
            "%.3g of the batch (gate %g); usage %d tokens = the pairs alone, and with top_n; "
            "%d requests %d at a time each got its own scores (worst %.3g)"
            % (max(worst.values()), RERANK_TOL, used, len(together), ARGS.concurrency,
               max(together)))


def check_rerank_limits(facts):
    """A pair past the context is refused, never truncated; malformed requests are 400s."""
    m = {"model": ARGS.model} if ARGS.model else {}
    status, j, data = post("/v1/rerank", dict(m, query=RERANK_QUERY,
                                              documents=["word " * 40000, RERANK_DOCS[1]]))
    facts["overlong"] = "%d %s" % (status, err_text(j, data)[:200])
    need(status == 400 and isinstance(j, dict) and "error" in j,
         "a 40000-token pair: %d %s" % (status, err_text(j, data)[:160]))
    for what, body in (("no documents", dict(m, query=RERANK_QUERY, documents=[])),
                       ("no query", dict(m, documents=RERANK_DOCS)),
                       ("a number as a document", dict(m, query=RERANK_QUERY, documents=[7]))):
        status, j, data = post("/v1/rerank", body)
        facts[what] = "%d %s" % (status, err_text(j, data)[:200])
        need(status == 400 and isinstance(j, dict) and "error" in j,
             "%s: %d %s" % (what, status, err_text(j, data)[:160]))
    every, _ = rerank({"query": RERANK_QUERY, "documents": RERANK_DOCS, "top_n": 0})
    need(len(every) == len(RERANK_DOCS), "top_n 0 returned %d results" % len(every))
    status, _, _ = request("GET", "/health", timeout=30)
    need(status == 200, "/health after the refusals: %d" % status)
    return ("a 40000-token pair refused (%s); no documents, no query and a number as a document "
            "400; top_n 0 returns all; /health 200 after" % facts["overlong"][:3])


def check_rerank_image(facts):
    """(capability image) A red picture ranks above a blue one for "red", and the reverse."""
    red = png_bytes(64, 64, lambda x, y: (220, 20, 20))
    blue = png_bytes(64, 64, lambda x, y: (20, 20, 220))
    docs = [{"content": [image_part(red)]}, {"content": [image_part(blue)]}]
    s_red, _ = rerank({"query": "a plain red square", "documents": docs})
    s_blue, _ = rerank({"query": "a plain blue square", "documents": docs})
    facts.update(red_query=s_red, blue_query=s_blue)
    need(s_red[0] > s_red[1], "for red, the red picture scores %.3f, the blue %.3f"
         % (s_red[0], s_red[1]))
    need(s_blue[1] > s_blue[0], "for blue, the blue picture scores %.3f, the red %.3f"
         % (s_blue[1], s_blue[0]))
    return "red query: red %.3f > blue %.3f; blue query: blue %.3f > red %.3f" % (
        s_red[0], s_red[1], s_blue[1], s_blue[0])


# ---------------------------------------------------------------- speech, transcription

SPEECH_TEXT = "The quick brown fox jumps over the lazy dog, and then it rests beside the river."


def save(name, data):
    if not ARGS.save:
        return None
    os.makedirs(ARGS.save, exist_ok=True)
    path = os.path.join(ARGS.save, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


SPEECH_REFUSAL = "not supported in this build"


def check_speech(facts):
    if ARGS.speech == "refused":
        return check_speech_refused(facts)
    return check_speech_served(facts)


def check_speech_refused(facts):
    """A Windows build of 0.3.0 refuses speech (ServerOptions::speech_refused): both POST routes
    answer 501 in the usual error shape, and the socket sends the refusal and closes 1008."""
    b = {"input": ARGS.speech_text, "response_format": "wav", "seed": 7}
    if ARGS.model:
        b["model"] = ARGS.model
    for path in ("/v1/audio/speech", "/audio/speech"):
        status, j, data = post(path, b, timeout=120)
        msg = err_text(j, data) if j is not None else "%d bytes" % len(data)
        facts[path] = "%d %s" % (status, msg[:200])
        need(status == 501, "%s: %d, not the 501 refusal: %s" % (path, status, msg[:160]))
        need(isinstance(j, dict) and isinstance(j.get("error"), dict),
             "%s: 501 without an error body: %s" % (path, msg[:160]))
        need(j["error"].get("type") == "not_implemented_error",
             "%s: error type %r, not not_implemented_error" % (path, j["error"].get("type")))
        need(SPEECH_REFUSAL in msg, "%s: the message does not say %r: %s"
             % (path, SPEECH_REFUSAL, msg[:160]))
    ws = WebSocket("/v1/audio/speech/stream", timeout=60)
    try:
        first = ws.recv()
        need(first[0] == "text", "the socket's first message was %r" % (first[:2],))
        ev = json.loads(first[1])
        facts["/v1/audio/speech/stream"] = ev
        need(ev.get("type") == "error" and SPEECH_REFUSAL in str(ev.get("message")),
             "the socket's first event is not the refusal: %s" % first[1][:200])
        last = ws.recv()
        need(last[0] == "close" and last[1] == 1008,
             "the socket was not closed with 1008: %r" % (last[:3],))
    finally:
        ws.sock.close()
    status, _, _ = request("GET", "/health", timeout=30)
    need(status == 200, "/health after the refusals: %d" % status)
    return ("refused: /v1/audio/speech and /audio/speech 501 not_implemented_error (%r); the socket "
            "an error event, then close 1008; /health 200" % SPEECH_REFUSAL)


def check_speech_served(facts):
    b = {"input": ARGS.speech_text, "response_format": "wav", "seed": 7}
    if ARGS.model:
        b["model"] = ARGS.model
    status, ctype, data = request("POST", "/v1/audio/speech", json.dumps(b).encode())
    need(status == 200, "/v1/audio/speech: %d %s" % (status, data[:200].decode("utf-8", "replace")))
    seconds, rate, channels, rms, peak = wav_stats(data)
    facts.update(seconds=seconds, rate=rate, channels=channels, rms=rms, peak=peak,
                 bytes=len(data), saved=save("speech.wav", data))
    words = len(ARGS.speech_text.split())
    # Speech runs about 2 to 3 words a second; a quarter to four times that is still speech.
    need(words / 12.0 <= seconds <= words * 1.6, "%.2f s of audio for %d words" % (seconds, words))
    need(rms > 0.01 and peak > 0.05, "near silence: RMS %.4f, peak %.4f" % (rms, peak))
    return "%.2f s for %d words, %d Hz x %d, RMS %.3f, peak %.3f%s" % (
        seconds, words, rate, channels, rms, peak,
        ", saved " + facts["saved"] if facts["saved"] else "")


def words_of(text):
    return ["".join(ch for ch in w.lower() if ch.isalnum()) for w in text.split()]


def check_transcription(facts):
    need(ARGS.audio and ARGS.audio_text, "needs --audio FILE and --audio-text WORDS")
    with open(ARGS.audio, "rb") as f:
        clip = f.read()
    fields = {"response_format": "json"}
    if ARGS.model:
        fields["model"] = ARGS.model
    status, j, data = post_form("/v1/audio/transcriptions", fields,
                                {"file": (os.path.basename(ARGS.audio), clip)})
    need(status == 200 and j, "/v1/audio/transcriptions: %d %s" % (status, err_text(j, data)))
    text = j.get("text") or ""
    facts["text"] = text
    want = [w for w in words_of(ARGS.audio_text) if w]
    got = set(words_of(text))
    hit = sum(1 for w in want if w in got)
    facts["words_found"] = "%d of %d" % (hit, len(want))
    need(hit >= 0.8 * len(want), "%d of %d words came back: %r" % (hit, len(want), text))
    return "%d of %d words: %r" % (hit, len(want), text)


def words_hit(text):
    """(words of --audio-text found in text, how many there are)"""
    want = [w for w in words_of(ARGS.audio_text) if w]
    got = set(words_of(text))
    return sum(1 for w in want if w in got), len(want)


def transcribe(clip, name, fields=None, expect=200):
    """POST /v1/audio/transcriptions. Returns (status, JSON or None, bytes)."""
    f = dict(fields or {})
    if ARGS.model:
        f["model"] = ARGS.model
    status, j, data = post_form("/v1/audio/transcriptions", f, {"file": (name, clip)})
    if expect == 200:
        need(status == 200, "/v1/audio/transcriptions %s: %d %s"
             % (f, status, err_text(j, data)[:160]))
    return status, j, data


def sse_events(data):
    out = []
    for block in data.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    out.append(json.loads(payload))
    return out


def check_transcription_forms(facts):
    """The same recording through every response_format and the stream gives one transcript; a
    repeat and requests in flight together are identical; what the route refuses is a 400."""
    need(ARGS.audio and ARGS.audio_text, "needs --audio FILE and --audio-text WORDS")
    with open(ARGS.audio, "rb") as f:
        clip = f.read()
    name = os.path.basename(ARGS.audio)
    _, j, _ = transcribe(clip, name)
    text = (j or {}).get("text") or ""
    need(text, "no text")
    usage = j.get("usage") or {}
    need(usage.get("input_tokens", 0) > 0 and usage.get("output_tokens", 0) > 0 and
         usage.get("total_tokens") == usage["input_tokens"] + usage["output_tokens"],
         "usage %s" % usage)
    _, j2, _ = transcribe(clip, name)
    need((j2 or {}).get("text") == text, "a repeat differs: %r, then %r" % (text, (j2 or {}).get("text")))
    _, _, plain = transcribe(clip, name, {"response_format": "text"})
    need(plain.decode("utf-8").strip() == text, "response_format text: %r" % plain[:200])
    _, v, _ = transcribe(clip, name, {"response_format": "verbose_json"})
    v = v or {}
    need(v.get("text") == text, "verbose_json's text: %r" % v.get("text"))
    need(isinstance(v.get("language"), str) and v["language"], "verbose_json names no language")
    segs = v.get("segments") or []
    need(segs and segs[0].get("start") == 0 and abs(segs[-1].get("end", -1) - v.get("duration", 0))
         < 1e-6, "verbose_json's segments: %s" % segs)
    facts["verbose"] = {"language": v.get("language"), "duration": v.get("duration"),
                        "segments": len(segs)}
    if clip[:4] == b"RIFF":
        seconds = wav_stats(clip)[0]
        need(abs(v.get("duration", 0) - seconds) < 0.05,
             "verbose_json's duration %.3f s; the WAV holds %.3f s" % (v.get("duration", 0), seconds))
    body, ctype = form_body(dict({"stream": "true"}, **({"model": ARGS.model} if ARGS.model else {})),
                            {"file": (name, clip)})
    status, rtype, data = request("POST", "/v1/audio/transcriptions", body, ctype)
    need(status == 200 and rtype.startswith("text/event-stream"),
         "stream: %d %s %r" % (status, rtype, data[:160]))
    ev = sse_events(data)
    deltas = [e.get("delta", "") for e in ev if e.get("type") == "transcript.text.delta"]
    done = [e for e in ev if e.get("type") == "transcript.text.done"]
    need(len(done) == 1 and ev[-1] is done[0], "stream: %d done events, last %s"
         % (len(done), ev[-1].get("type") if ev else None))
    need(len(deltas) > 1 and "".join(deltas) == done[0].get("text") == text,
         "stream: %d deltas joined %r, done %r, whole %r"
         % (len(deltas), "".join(deltas), done[0].get("text"), text))
    need(done[0].get("usage") == usage, "stream's usage %s, whole's %s" % (done[0].get("usage"), usage))
    facts["stream_deltas"] = len(deltas)
    with ThreadPoolExecutor(8) as pool:
        together = list(pool.map(lambda _: (transcribe(clip, name)[1] or {}).get("text"), range(8)))
    need(all(t == text for t in together), "8 at a time: %s" % [t for t in together if t != text])
    refused = {}
    for what, fields in (("srt", {"response_format": "srt"}),
                         ("temperature 0.5", {"temperature": "0.5"}),
                         ("language xx", {"language": "xx"})):
        status, jr, data = transcribe(clip, name, fields, expect=None)
        refused[what] = "%d %s" % (status, err_text(jr, data)[:160])
        need(status == 400 and isinstance(jr, dict) and "error" in jr, "%s: %s" % (what, refused[what]))
    status, jr, data = transcribe(b"this is not a recording" * 50, "clip.wav", expect=None)
    refused["not audio"] = "%d %s" % (status, err_text(jr, data)[:160])
    need(status == 400 and isinstance(jr, dict) and "error" in jr, "not audio: %s" % refused["not audio"])
    facts["refused"] = refused
    status, jl, _ = transcribe(clip, name, {"language": "en"})
    hit, of = words_hit((jl or {}).get("text") or "")
    need(hit >= 0.8 * of, "language en: %d of %d words: %r" % (hit, of, (jl or {}).get("text")))
    status, _, _ = request("GET", "/health", timeout=30)
    need(status == 200, "/health after the refusals: %d" % status)
    return ("json, text, verbose_json (%s, %.2f s) and the stream (%d deltas) give one transcript; "
            "a repeat and 8 at a time identical; language en %d of %d words; srt, temperature, an "
            "unknown language and a file that is not audio are 400"
            % (v.get("language"), v.get("duration", 0), len(deltas), hit, of))


def check_transcription_alt(facts):
    """(--audio-alt FILE ...) Other encodings of the --audio recording -- mp3, flac, ogg, m4a, ... --
    each give most of --audio-text's words, as the original does."""
    need(ARGS.audio_alt and ARGS.audio_text, "needs --audio-alt FILE and --audio-text WORDS")
    main = ""
    if ARGS.audio:
        with open(ARGS.audio, "rb") as f:
            main = (transcribe(f.read(), os.path.basename(ARGS.audio))[1] or {}).get("text", "")
    same, lines = 0, []
    for path in ARGS.audio_alt:
        with open(path, "rb") as f:
            _, j, _ = transcribe(f.read(), os.path.basename(path))
        text = (j or {}).get("text") or ""
        hit, of = words_hit(text)
        facts[os.path.basename(path)] = text
        need(hit >= 0.8 * of, "%s: %d of %d words: %r" % (os.path.basename(path), hit, of, text))
        same += text == main
        lines.append("%s %d/%d" % (os.path.splitext(path)[1].lstrip("."), hit, of))
    return "%s; %d of %d identical to the original's transcript" % (
        ", ".join(lines), same, len(ARGS.audio_alt))


# ---------------------------------------------------------------- realtime

class WebSocket:
    """A client WebSocket (RFC 6455) over a plain socket: enough for /v1/realtime."""

    def __init__(self, path, timeout=120):
        u = urllib.parse.urlparse(ARGS.url)
        port = u.port or 80
        self.sock = socket.create_connection((u.hostname, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        head = ["GET %s HTTP/1.1" % path, "Host: %s:%d" % (u.hostname, port),
                "Upgrade: websocket", "Connection: Upgrade", "Sec-WebSocket-Key: " + key,
                "Sec-WebSocket-Version: 13"]
        if ARGS.key:
            head.append("Authorization: Bearer " + ARGS.key)
        self.sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode())
        got = b""
        while b"\r\n\r\n" not in got:
            chunk = self.sock.recv(4096)
            need(chunk, "the socket closed during the handshake: %r" % got[:200])
            got += chunk
        got, self.buf = got.split(b"\r\n\r\n", 1)
        lines = got.decode("latin-1").split("\r\n")
        need(lines[0].split()[1:2] == ["101"], "the upgrade was answered %r" % lines[0])
        hdr = {k.strip().lower(): v.strip() for k, v in
               (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        need(hdr.get("sec-websocket-accept") == accept, "a wrong Sec-WebSocket-Accept")
        self.lock = threading.Lock()

    def send(self, opcode, payload):
        n = len(payload)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        if n:
            m = (mask * (n // 4 + 1))[:n]
            payload = (int.from_bytes(payload, "big") ^ int.from_bytes(m, "big")).to_bytes(n, "big")
        with self.lock:
            self.sock.sendall(head + mask + payload)

    def send_json(self, obj):
        self.send(1, json.dumps(obj).encode())

    def _exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("the server closed the connection")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        """The next message: ("text", str), ("binary", bytes) or ("close", code, reason)."""
        parts, kind = [], None
        while True:
            b0, b1 = self._exact(2)
            op, n = b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._exact(8))[0]
            mask = self._exact(4) if b1 & 0x80 else None
            data = self._exact(n)
            if mask:
                data = bytes(c ^ mask[i & 3] for i, c in enumerate(data))
            if op == 9:
                self.send(10, data)
                continue
            if op == 10:
                continue
            if op == 8:
                code = struct.unpack(">H", data[:2])[0] if len(data) >= 2 else None
                return ("close", code, data[2:].decode("utf-8", "replace"))
            if op in (1, 2):
                kind = op
            parts.append(data)
            if b0 & 0x80:
                data = b"".join(parts)
                return ("text", data.decode("utf-8")) if kind == 1 else ("binary", data)

    def close(self, code=1000):
        try:
            self.send(8, struct.pack(">H", code))
        except OSError:
            pass


class RealtimeSession:
    """A /v1/realtime session: a reader thread files every event the server sends, in order."""

    def __init__(self):
        self.ws = WebSocket("/v1/realtime")
        self.events, self.closed = [], None
        self.cv = threading.Condition()
        self.t0 = time.monotonic()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                m = self.ws.recv()
            except Exception as e:
                m = ("close", None, "%s: %s" % (type(e).__name__, e))
            with self.cv:
                if m[0] == "close":
                    self.closed = m[1:]
                elif m[0] == "text":
                    ev = json.loads(m[1])
                    ev["_t"] = time.monotonic() - self.t0
                    self.events.append(ev)
                self.cv.notify_all()
            if m[0] == "close":
                return

    def wait(self, pred, what, timeout=120, start=0):
        """The first event from index start on that pred accepts; fails on a timeout or a close."""
        end = time.monotonic() + timeout
        with self.cv:
            while True:
                for i in range(start, len(self.events)):
                    if pred(self.events[i]):
                        return i, self.events[i]
                need(self.closed is None, "the socket closed (%s) waiting for %s; events %s"
                     % (self.closed, what, [e.get("type") for e in self.events[start:]][-8:]))
                left = end - time.monotonic()
                need(left > 0, "no %s within %d s; events since: %s"
                     % (what, timeout, [e.get("type") for e in self.events[start:]][-8:]))
                self.cv.wait(left)

    def of(self, kind, start=0):
        with self.cv:
            return [e for e in self.events[start:] if e.get("type") == kind]


def wav_pcm(data):
    """(16-bit mono PCM bytes, rate) of a WAV."""
    with wave.open(io.BytesIO(data), "rb") as w:
        need(w.getnchannels() == 1 and w.getsampwidth() == 2,
             "a realtime session takes 16-bit mono PCM; the WAV is %d channel(s) x %d bits"
             % (w.getnchannels(), 8 * w.getsampwidth()))
        return w.readframes(w.getnframes()), w.getframerate()


def realtime_turn(s, pcm, rate, paced, binary=False):
    """One turn: the audio in 100 ms appends (at real time when paced), then a commit. Returns
    (the completed event, the deltas, the deltas that came before the commit, seconds from the
    commit to the transcript). Each delta's "_t" is rebased to seconds from the turn's first audio."""
    start = len(s.events)
    step = rate // 10 * 2
    t0 = time.monotonic()
    for k, off in enumerate(range(0, len(pcm), step)):
        if paced:
            delay = t0 + k * 0.1 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        chunk = pcm[off:off + step]
        if binary:
            s.ws.send(2, chunk)
        else:
            s.ws.send_json({"type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode()})
    before = len(s.of("conversation.item.input_audio_transcription.delta", start))
    t_commit = time.monotonic() - s.t0
    s.ws.send_json({"type": "input_audio_buffer.commit"})
    i, committed = s.wait(lambda e: e.get("type") in ("input_audio_buffer.committed", "error"),
                          "input_audio_buffer.committed", 60, start)
    need(committed["type"] == "input_audio_buffer.committed", "the commit: %s" % committed)
    item = committed.get("item_id")
    _, done = s.wait(lambda e: e.get("type") in (
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.failed", "error"),
        "the turn's transcript", 120, start)
    need(done["type"] == "conversation.item.input_audio_transcription.completed",
         "the turn: %s" % done)
    need(done.get("item_id") == item, "the transcript is for item %s; the commit was %s"
         % (done.get("item_id"), item))
    deltas = [dict(e, _t=e["_t"] - (t0 - s.t0))
              for e in s.of("conversation.item.input_audio_transcription.delta", start)
              if e.get("item_id") == item]
    return done, deltas, before, done["_t"] - t_commit


def check_realtime(facts):
    """/v1/realtime, OpenAI's live transcription over a WebSocket: --audio (a 16-bit mono WAV) streamed
    at real time gives words as they settle, and the turn's transcript matches the upload route's;
    the same audio as binary frames gives it again; a bad update is an error event, not a close."""
    need(ARGS.audio and ARGS.audio_text, "needs --audio FILE (a WAV) and --audio-text WORDS")
    with open(ARGS.audio, "rb") as f:
        clip = f.read()
    need(clip[:4] == b"RIFF", "--audio must be a WAV for the realtime check")
    pcm, rate = wav_pcm(clip)
    upload = (transcribe(clip, os.path.basename(ARGS.audio))[1] or {}).get("text") or ""
    s = RealtimeSession()
    try:
        _, created = s.wait(lambda e: True, "session.created", 30)
        need(created.get("type") == "session.created", "the first event is %s" % created)
        s.ws.send_json({"type": "session.update", "session": {
            "type": "transcription",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 12345}}}}})
        _, bad = s.wait(lambda e: e.get("type") in ("error", "session.updated"),
                        "the bad update's answer", 30)
        need(bad["type"] == "error", "a 12345 Hz update was accepted: %s" % bad)
        facts["bad_update"] = bad.get("error")
        n = len(s.events)
        tr = {"model": ARGS.model} if ARGS.model else {}
        s.ws.send_json({"type": "session.update", "session": {
            "type": "transcription",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": rate},
                                "transcription": tr, "turn_detection": None}}}})
        _, upd = s.wait(lambda e: e.get("type") in ("error", "session.updated"),
                        "session.updated", 30, n)
        need(upd["type"] == "session.updated", "the update: %s" % upd)
        fmt = (((upd.get("session") or {}).get("audio") or {}).get("input") or {}).get("format")
        need((fmt or {}).get("rate") == rate, "session.updated says the format is %s" % fmt)
        done, deltas, before, lag = realtime_turn(s, pcm, rate, paced=True)
        text = done.get("transcript") or ""
        joined = "".join(d.get("delta", "") for d in deltas)
        hit, of = words_hit(text)
        facts["turn1"] = {"transcript": text, "deltas": len(deltas), "before_commit": before,
                          "first_delta_s": deltas[0]["_t"] if deltas else None,
                          "commit_to_transcript_s": lag, "usage": done.get("usage"),
                          "upload": upload}
        need(hit >= 0.8 * of, "turn 1: %d of %d words: %r" % (hit, of, text))
        need(before > 0, "streamed at real time, no word settled before the commit")
        need(text.startswith(joined), "the deltas joined, %r, are not the start of the transcript "
             "%r" % (joined, text))
        need(text == upload, "the live transcript %r differs from the upload route's %r"
             % (text, upload))
        u = done.get("usage") or {}
        need(u.get("input_tokens", 0) > 0 and u.get("output_tokens", 0) > 0, "usage %s" % u)
        done2, _, _, lag2 = realtime_turn(s, pcm, rate, paced=False, binary=True)
        need(done2.get("transcript") == text, "turn 2 (binary frames, unpaced): %r; turn 1 %r"
             % (done2.get("transcript"), text))
        facts["turn2_commit_to_transcript_s"] = lag2
        n = len(s.events)
        s.ws.send_json({"type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[:rate]).decode()})
        s.ws.send_json({"type": "input_audio_buffer.clear"})
        s.wait(lambda e: e.get("type") == "input_audio_buffer.cleared", "the clear", 30, n)
        errors = s.of("error")
        need(len(errors) == 1, "error events beyond the bad update: %s" % errors[1:])
    finally:
        s.ws.close(1000)
    end = time.monotonic() + 10
    while s.closed is None and time.monotonic() < end:
        time.sleep(0.05)
    facts["close"] = s.closed
    need(s.closed is not None and s.closed[0] == 1000, "the close was answered %s" % (s.closed,))
    status, _, _ = request("GET", "/health", timeout=30)
    need(status == 200, "/health after the session: %d" % status)
    return ("streamed at real time: %d deltas (%d before the commit, the first at %.2f s), the "
            "transcript %.2f s after the commit and equal to the upload route's: %r; binary "
            "frames unpaced gave the same; a bad rate an error event; clear and close answered"
            % (len(deltas), before, deltas[0]["_t"] if deltas else -1, lag, text))


# ---------------------------------------------------------------- images

def check_images(facts):
    w, h = [int(x) for x in ARGS.image_size.split("x")]
    b = {"prompt": "a lighthouse at dusk, oil painting", "size": ARGS.image_size, "seed": 7,
         "response_format": "b64_json"}
    if ARGS.model:
        b["model"] = ARGS.model
    t0 = time.time()
    status, j, data = post("/v1/images/generations", b, timeout=1800)
    need(status == 200 and j, "/v1/images/generations: %d %s" % (status, err_text(j, data)))
    took = time.time() - t0
    rows = j.get("data")
    need(isinstance(rows, list) and len(rows) == 1 and rows[0].get("b64_json"), "no b64_json")
    png = base64.b64decode(rows[0]["b64_json"])
    pw, ph, channels, pixels = png_decode(png)
    facts.update(width=pw, height=ph, channels=channels, bytes=len(png), seconds=took,
                 saved=save("image.png", png))
    need((pw, ph) == (w, h), "asked %dx%d, got %dx%d" % (w, h, pw, ph))
    mean = sum(pixels) / len(pixels)
    spread = math.sqrt(sum((p - mean) ** 2 for p in pixels[::7]) / len(pixels[::7]))
    facts["spread"] = spread
    need(spread > 8, "the picture is nearly one flat colour (spread %.2f)" % spread)
    status, j2, data = post("/v1/images/generations", b, timeout=1800)
    need(status == 200 and j2, "the second picture: %d %s" % (status, err_text(j2, data)))
    need(base64.b64decode(j2["data"][0]["b64_json"]) == png, "the same seed gave another picture")
    return "%dx%d PNG, %d channels, %d bytes, spread %.1f, %.1f s; the same seed repeats it" % (
        pw, ph, channels, len(png), spread, took)


def jpeg_size(data):
    """(width, height) from a JPEG's frame header; the markers before it are walked by their
    lengths and the file must end in EOI, so a cut-short or damaged file fails here."""
    need(data[:3] == b"\xff\xd8\xff", "not a JPEG (starts %r)" % data[:4])
    need(data[-2:] == b"\xff\xd9", "JPEG without its end marker")
    pos = 2
    while pos + 4 <= len(data):
        need(data[pos] == 0xFF, "JPEG marker expected at %d" % pos)
        kind = data[pos + 1]
        n, = struct.unpack(">H", data[pos + 2:pos + 4])
        if kind in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[pos + 5:pos + 9])
            return w, h
        need(kind != 0xDA, "JPEG scan before any frame header")
        pos += 2 + n
    raise Fail("JPEG without a frame header")


def webp_size(data):
    """(width, height) from a WebP's first chunk (VP8, VP8L or VP8X); the RIFF length must be the
    file's."""
    need(data[:4] == b"RIFF" and data[8:12] == b"WEBP", "not a WebP (starts %r)" % data[:12])
    n, = struct.unpack("<I", data[4:8])
    need(n + 8 == len(data), "WebP's RIFF says %d bytes, the file has %d" % (n + 8, len(data)))
    kind = data[12:16]
    if kind == b"VP8 ":
        need(data[23:26] == b"\x9d\x01\x2a", "VP8 frame without its start code")
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if kind == b"VP8L":
        need(data[20] == 0x2F, "VP8L without its signature")
        b, = struct.unpack("<I", data[21:25])
        return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
    if kind == b"VP8X":
        return (int.from_bytes(data[24:27], "little") + 1,
                int.from_bytes(data[27:30], "little") + 1)
    raise Fail("WebP chunk %r not read here" % kind)


def images_post(body, what):
    """A generation that must succeed: (response JSON, [(picture bytes, row)], seconds)."""
    if ARGS.model:
        body = dict(body, model=ARGS.model)
    t0 = time.time()
    status, j, data = post("/v1/images/generations", body, timeout=1800)
    took = time.time() - t0
    need(status == 200 and isinstance(j, dict), "%s: %d %s" % (what, status, err_text(j, data)))
    rows = j.get("data")
    need(isinstance(rows, list) and rows and all(isinstance(r, dict) and r.get("b64_json")
                                                for r in rows), "%s: no b64_json rows" % what)
    return j, [(base64.b64decode(r["b64_json"]), r) for r in rows], took


IMAGE_PROMPT = "a lighthouse at dusk, oil painting"


def check_images_forms(facts):
    """JPEG and WebP of the size asked, each saying so in output_format; the default form is
    b64_json; response_format url is either served (a URL that answers with the picture) or refused
    with a 400 naming response_format."""
    w, h = [int(x) for x in ARGS.image_size.split("x")]
    out = []
    for fmt, size_of in (("jpeg", jpeg_size), ("webp", webp_size)):
        j, pics, took = images_post({"prompt": IMAGE_PROMPT, "size": ARGS.image_size, "seed": 7,
                                     "output_format": fmt}, fmt)
        need(len(pics) == 1, "%s: %d pictures for n 1" % (fmt, len(pics)))
        data = pics[0][0]
        pw, ph = size_of(data)
        facts[fmt] = {"width": pw, "height": ph, "bytes": len(data), "seconds": took,
                      "output_format": j.get("output_format"),
                      "saved": save("image." + ("jpg" if fmt == "jpeg" else fmt), data)}
        need((pw, ph) == (w, h), "%s: asked %dx%d, got %dx%d" % (fmt, w, h, pw, ph))
        need(j.get("output_format") in (None, fmt), "%s: the response says output_format %r"
             % (fmt, j.get("output_format")))
        need(len(data) > 2000, "%s: %d bytes is no picture" % (fmt, len(data)))
        out.append("%s %dx%d %d bytes" % (fmt, pw, ph, len(data)))
    m = {"model": ARGS.model} if ARGS.model else {}
    status, j, data = post("/v1/images/generations",
                           dict(m, prompt=IMAGE_PROMPT, size=ARGS.image_size, seed=7,
                                response_format="url"), timeout=1800)
    msg = err_text(j, data)
    facts["url"] = "%d %s" % (status, msg[:200])
    if status == 200:
        url = (j.get("data") or [{}])[0].get("url") or ""
        need(url.startswith("http"), "response_format url: 200 without a URL")
        u = urllib.parse.urlparse(url)
        c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)
        c.request("GET", u.path + ("?" + u.query if u.query else ""))
        r = c.getresponse()
        got = r.read()
        need(r.status == 200 and png_decode(got)[:2] == (w, h), "the URL did not give the picture")
        out.append("url served")
    else:
        need(status == 400 and isinstance(j, dict) and "error" in j and "response_format" in msg,
             "response_format url: %d %s" % (status, msg[:160]))
        out.append("url refused 400 (%s)" % msg[:80])
    j, pics, _ = images_post({"prompt": IMAGE_PROMPT, "size": ARGS.image_size, "seed": 7},
                             "no response_format")
    png_decode(pics[0][0])
    out.append("b64_json PNG by default")
    return "; ".join(out)


def check_images_sizes(facts):
    """A second, non-square size comes back as asked; a size off the model's grid comes back
    rounded down to it, never up, and the response's size names the picture's."""
    out = []
    for size in (ARGS.image_size2, "1000x700"):
        aw, ah = [int(x) for x in size.split("x")]
        j, pics, took = images_post({"prompt": IMAGE_PROMPT, "size": size, "seed": 7}, size)
        pw, ph, channels, pixels = png_decode(pics[0][0])
        facts[size] = {"width": pw, "height": ph, "response_size": j.get("size"), "seconds": took,
                       "saved": save("image-%s.png" % size, pics[0][0])}
        if j.get("size") is not None:
            need(j["size"] == "%dx%d" % (pw, ph), "%s: the response says %s, the PNG is %dx%d"
                 % (size, j["size"], pw, ph))
        if size == ARGS.image_size2:
            need((pw, ph) == (aw, ah), "asked %s, got %dx%d" % (size, pw, ph))
        else:   # every model here rounds to a grid of 8 to 64 pixels
            need(aw - 64 < pw <= aw and ah - 64 < ph <= ah, "asked %s, got %dx%d" % (size, pw, ph))
        mean = sum(pixels) / len(pixels)
        spread = math.sqrt(sum((p - mean) ** 2 for p in pixels[::7]) / len(pixels[::7]))
        need(spread > 8, "%s: nearly one flat colour (spread %.2f)" % (size, spread))
        out.append("%s -> %dx%d in %.1f s" % (size, pw, ph, took))
    return "; ".join(out)


def check_images_n(facts):
    """n 2 gives two different pictures, each with its seed; picture i is seed + i's own picture,
    the same bytes as asked for alone; pictures of two sizes in flight together are each the bytes
    they are alone."""
    _, solo7, _ = images_post({"prompt": IMAGE_PROMPT, "size": ARGS.image_size, "seed": 7}, "seed 7")
    _, solo8, _ = images_post({"prompt": IMAGE_PROMPT, "size": ARGS.image_size, "seed": 8}, "seed 8")
    j, pics, took = images_post({"prompt": IMAGE_PROMPT, "size": ARGS.image_size, "seed": 7, "n": 2},
                                "n 2")
    need(len(pics) == 2, "n 2 gave %d pictures" % len(pics))
    seeds = [r.get("seed") for _, r in pics]
    facts.update(seeds=seeds, seconds=took,
                 saved=[save("image-n2-%d.png" % i, p) for i, (p, _) in enumerate(pics)])
    for p, _ in pics:
        png_decode(p)
    need(pics[0][0] != pics[1][0], "n 2 gave the same picture twice")
    need(seeds == [7, 8] or seeds == [None, None], "n 2 at seed 7 reports seeds %s" % seeds)
    need(pics[0][0] == solo7[0][0], "n 2's first picture is not seed 7's picture alone")
    need(pics[1][0] == solo8[0][0], "n 2's second picture is not seed 8's picture alone")
    small = {"prompt": IMAGE_PROMPT, "size": ARGS.image_size2, "seed": 7}
    _, solo_small, _ = images_post(small, ARGS.image_size2)
    with ThreadPoolExecutor(2) as ex:
        a = ex.submit(images_post, {"prompt": IMAGE_PROMPT, "size": ARGS.image_size, "seed": 7},
                      "together, " + ARGS.image_size)
        b = ex.submit(images_post, small, "together, " + ARGS.image_size2)
        ra, rb = a.result(), b.result()
    need(ra[1][0][0] == solo7[0][0], "%s in flight beside %s changed" % (ARGS.image_size,
                                                                        ARGS.image_size2))
    need(rb[1][0][0] == solo_small[0][0], "%s in flight beside %s changed" % (ARGS.image_size2,
                                                                             ARGS.image_size))
    return ("n 2: two pictures, seeds %s, each the bytes of its seed alone, %.1f s; %s and %s in "
            "flight together give their solo bytes" % (seeds, took, ARGS.image_size,
                                                       ARGS.image_size2))


def check_images_limits(facts):
    """Malformed and impossible requests are 400s with an error body, not a picture, a 500 or a
    hang: sizes that are not WxH, a side of zero or under the smallest, a picture past any model's
    area, no prompt, a prompt that is not text or longer than the model reads, n out of range, an
    output_format or response_format that does not exist."""
    m = {"model": ARGS.model} if ARGS.model else {}
    base = dict(m, prompt="a red apple", size="512x512")
    long_prompt = " ".join(["a red apple on a wooden table"] * 200)   # 1400 words
    probes = [
        ("size abc", dict(base, size="abc")),
        ("size 512", dict(base, size="512")),
        ("size 0x512", dict(base, size="0x512")),
        ("size -512x512", dict(base, size="-512x512")),
        ("size 16x16", dict(base, size="16x16")),
        ("size 8192x8192", dict(base, size="8192x8192")),
        ("size 512", dict(base, size=512)),
        ("no prompt", {k: v for k, v in base.items() if k != "prompt"}),
        ("prompt a number", dict(base, prompt=5)),
        ("prompt 1400 words", dict(base, prompt=long_prompt)),
        ("n 0", dict(base, n=0)),
        ("n 17", dict(base, n=17)),
        ("n 1.5", dict(base, n=1.5)),
        ("output_format gif", dict(base, output_format="gif")),
        ("response_format bmp", dict(base, response_format="bmp")),
    ]
    for what, body in probes:
        status, j, data = post("/v1/images/generations", body, timeout=600)
        msg = err_text(j, data)
        facts[what] = "%d %s" % (status, msg[:200])
        need(status == 400 and isinstance(j, dict) and isinstance(j.get("error"), dict),
             "%s: %d %s" % (what, status, msg[:160]))
    # Recorded, not judged: OpenAI requires a prompt but says nothing of an empty one.
    status_empty, j, data = post("/v1/images/generations", dict(base, prompt=""), timeout=600)
    facts["empty prompt"] = "%d %s" % (status_empty, "a picture" if status_empty == 200
                                       else err_text(j, data)[:120])
    status, _, _ = request("GET", "/health", timeout=30)
    need(status == 200, "/health after the refusals: %d" % status)
    return "%d malformed requests each a 400 with an error body; an empty prompt %d; /health 200" % (
        len(probes), status_empty)


# ---------------------------------------------------------------- refusals

def check_refusals(facts, caps):
    """What this model does not serve answers 501 with an error body -- not a 404, a 500 or a hang."""
    m = {"model": ARGS.model} if ARGS.model else {}
    probes = []
    if not caps.get("embedding"):
        probes.append(("/v1/embeddings", dict(m, input="hello")))
    if not caps.get("rerank"):
        probes.append(("/v1/rerank", dict(m, query="q", documents=["a", "b"])))
    if not caps.get("chat"):
        probes.append(("/v1/chat/completions",
                       dict(m, messages=[{"role": "user", "content": "hi"}], max_tokens=4)))
    if not caps.get("image_generation"):
        probes.append(("/v1/images/generations", dict(m, prompt="a red apple", size="512x512")))
    if not caps.get("audio_generation"):
        probes.append(("/v1/audio/speech", dict(m, input="hello")))
    if not caps.get("video_generation"):
        probes.append(("/v1/videos", dict(m, prompt="a red apple")))
    seen = []
    for path, body in probes:
        status, j, data = post(path, body, timeout=120)
        msg = err_text(j, data)
        facts[path] = "%d %s" % (status, msg[:200])
        need(status == 501 and isinstance(j, dict) and "error" in j,
             "%s: %d %s" % (path, status, msg[:160]))
        seen.append(path)
    if not ARGS.transcribes:   # no capability names transcription; the flag says the model does
        status, j, data = post_form("/v1/audio/transcriptions", dict(m),
                                    {"file": ("tone.wav", wav_bytes(tone(440, 0.5, 0.3)))}, 120)
        msg = err_text(j, data)
        facts["/v1/audio/transcriptions"] = "%d %s" % (status, msg[:200])
        need(status == 501 and isinstance(j, dict) and "error" in j,
             "/v1/audio/transcriptions: %d %s" % (status, msg[:160]))
        seen.append("/v1/audio/transcriptions")
    status, _, _ = request("GET", "/health", timeout=30)
    need(status == 200, "/health after the refusals: %d" % status)
    return "501 with an error body from %s; /health 200 after" % ", ".join(seen)


# ---------------------------------------------------------------- selftest

def check_selftest(facts):
    png = png_bytes(37, 21, lambda x, y: (x * 6 & 255, y * 11 & 255, (x + y) & 255))
    w, h, ch, px = png_decode(png)
    need((w, h, ch) == (37, 21, 3), "PNG round trip: %dx%d x%d" % (w, h, ch))
    need(px[3 * (5 * 37 + 9):3 * (5 * 37 + 9) + 3] == bytes((54, 55, 14)), "PNG pixel (9, 5)")
    bad = bytearray(png)
    bad[60] ^= 1
    try:
        png_decode(bytes(bad))
        raise Fail("a damaged PNG decoded")
    except Fail as e:
        need("decoded" not in str(e), str(e))
    except zlib.error:
        pass
    seconds, rate, channels, rms, peak = wav_stats(wav_bytes(tone(440, 1.5, 0.5, 24000), 24000))
    need(abs(seconds - 1.5) < 1e-3 and rate == 24000 and channels == 1, "WAV length %.3f" % seconds)
    need(abs(rms - 0.5 / math.sqrt(2)) < 0.01 and abs(peak - 0.5) < 0.01,
         "WAV level: RMS %.3f, peak %.3f" % (rms, peak))
    _, _, _, rms0, _ = wav_stats(wav_bytes([0.0] * 16000))
    need(rms0 == 0.0, "silence has RMS %.4f" % rms0)
    need(abs(cos([1, 0, 0], [1, 1, 0]) - 1 / math.sqrt(2)) < 1e-9, "cosine")
    return "PNG write/read and a damaged one refused; WAV length and level; cosine"


# ---------------------------------------------------------------- main

def model_card():
    status, _, data = request("GET", "/v1/models", timeout=60)
    need(status == 200, "/v1/models: %d" % status)
    cards = json.loads(data.decode("utf-8")).get("data") or []
    need(cards, "/v1/models lists no model")
    for c in cards:
        if c.get("id") == ARGS.model:
            return c
    need(not ARGS.model, "/v1/models has no model %r (it has %s)"
         % (ARGS.model, [c.get("id") for c in cards]))
    return cards[0]


def main():
    global ARGS, LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8120")
    ap.add_argument("--model", default="", help="the served model's name (default: the first)")
    ap.add_argument("--key", default="")
    ap.add_argument("--out", default="", help="a JSON-lines record of every check")
    ap.add_argument("--save", default="", help="a directory for what the server made (WAV, PNG)")
    ap.add_argument("--only", default="", help="comma-separated check names, instead of the "
                    "capabilities' choice")
    ap.add_argument("--query-prompt", default="", help="prompt_name for queries (query)")
    ap.add_argument("--doc-prompt", default="", help="prompt_name for documents (document)")
    ap.add_argument("--audio", default="", help="a recording, for transcription and embeddings")
    ap.add_argument("--audio-text", default="", help="what the recording says")
    ap.add_argument("--audio-alt", action="append", default=[],
                    help="the --audio recording in another encoding (mp3, flac, ogg, m4a, ...), "
                    "for transcription_alt; repeat it for each")
    ap.add_argument("--audio-other", default="", help="an unrelated text, for the audio embedding")
    ap.add_argument("--video", default="", help="a clip for embeddings_video")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="embedding requests in flight together")
    ap.add_argument("--dump", default="", help="write embeddings_compare's vectors here (JSON)")
    ap.add_argument("--compare", default="", help="hold them against an earlier --dump")
    ap.add_argument("--compare-min", type=float, default=0.999,
                    help="the lowest cosine --compare accepts")
    ap.add_argument("--speech-text", default=SPEECH_TEXT)
    ap.add_argument("--speech", choices=("refused", "served"),
                    default="refused" if sys.platform == "win32" else "served",
                    help="what the speech check expects: refused (the default on Windows, where "
                    "Brilliance 0.3.0 builds answer 501) or served (a WAV; a build that serves "
                    "speech)")
    ap.add_argument("--image-size", default="512x512")
    ap.add_argument("--image-size2", default="768x512",
                    help="a second, non-square size for images_sizes and images_n")
    ap.add_argument("--transcribes", action="store_true",
                    help="the model serves /v1/audio/transcriptions (no capability names it)")
    ARGS = ap.parse_args()
    if ARGS.out:
        os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
        LOG = open(ARGS.out, "a", encoding="utf-8")

    only = [s for s in ARGS.only.split(",") if s]
    caps = {}
    if only != ["selftest"]:
        try:
            card = model_card()
        except Exception as e:
            record("model", False, "%s: %s" % (type(e).__name__, e))
            return 1
        caps = card.get("capabilities") or {}
        if not ARGS.model:
            ARGS.model = card.get("id") or ""
        record("model", True, "%s; capabilities: %s" % (
            ARGS.model, ", ".join(sorted(k for k, v in caps.items() if v is True)) or "none"),
            {"card": card})

    table = {
        "selftest": check_selftest,
        "embeddings": check_embeddings,
        "embeddings_image": check_embeddings_image,
        "embeddings_audio": check_embeddings_audio,
        "embeddings_video": check_embeddings_video,
        "embeddings_compare": lambda facts: check_embeddings_compare(facts, caps),
        "rerank": check_rerank,
        "rerank_batch": check_rerank_batch,
        "rerank_limits": check_rerank_limits,
        "rerank_image": check_rerank_image,
        "speech": check_speech,
        "transcription": check_transcription,
        "transcription_forms": check_transcription_forms,
        "transcription_alt": check_transcription_alt,
        "realtime": check_realtime,
        "images": check_images,
        "images_forms": check_images_forms,
        "images_sizes": check_images_sizes,
        "images_n": check_images_n,
        "images_limits": check_images_limits,
        "refusals": lambda facts: check_refusals(facts, caps),
    }
    if not only:
        only = ["selftest"]
        if caps.get("embedding"):
            only.append("embeddings")
            if caps.get("image"):
                only.append("embeddings_image")
            if caps.get("audio"):
                only.append("embeddings_audio")
            if caps.get("video") and ARGS.video:
                only.append("embeddings_video")
            if ARGS.dump or ARGS.compare:
                only.append("embeddings_compare")
        if caps.get("rerank"):
            only += ["rerank", "rerank_batch", "rerank_limits"]
            if caps.get("image"):
                only.append("rerank_image")
        if caps.get("audio_generation"):
            only.append("speech")
        if ARGS.transcribes:
            only += ["transcription", "transcription_forms"]
            if ARGS.audio_alt:
                only.append("transcription_alt")
            if ARGS.audio.lower().endswith(".wav"):
                only.append("realtime")
        if caps.get("image_generation"):
            only += ["images", "images_forms", "images_sizes", "images_n", "images_limits"]
        only.append("refusals")
    for name in only:
        if name not in table:
            record(name, False, "no such check (there are: %s)" % ", ".join(table))
            continue
        run(name, table[name])
    ran = [ok for name, ok in RESULTS if name != "model"]
    print("%d of %d checks passed" % (sum(ran), len(ran)), flush=True)
    return 0 if ran and all(ok for _, ok in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
