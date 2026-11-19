#!/usr/bin/env python3
"""Brilliance: the API qualification of a served model, against a live server.

Each check sends real requests and judges the answer; the run ends with one line a check and exits 0
only when every selected check passed. It needs only the standard library, so the toolchain's own
CPython runs it (`.toolchain\\python\\python.exe -I win\\tools\\qualify_api.py ...`).

What it covers, by check name:
  chat, chat_stream        Chat Completions, whole and streamed; the stream's text equals the
                           whole answer's (greedy), and ends in [DONE]
  responses, responses_stream, messages, messages_stream
                           the Responses and Messages APIs, whole and streamed
  tools_typed              a tool whose parameters are an integer, a number, a boolean, an enum,
                           an array and an object: the arguments parse and have those types
  tools_multiline          a string argument that holds newlines arrives with them
  tools_empty              a tool with no parameters is called with {} (or nothing)
  tools_multiple           two independent lookups: two calls, in one turn or over two
  tool_roundtrip_chat, tool_roundtrip_responses, tool_roundtrip_messages
                           a tool result sent back is used in the answer
  cancel_prefill, cancel_decode
                           a client that disconnects during prefill or decode; the server stays
                           healthy and answers the next request
  malformed                bad JSON, an unknown model, missing fields, wrong types: 4xx with an
                           error body, never a 5xx or a hang, and the server answers afterwards
  issue4                   the Codeberg issue #4 regression: asked for the free disk space with a
                           shell tool, the model calls it for that, takes the mocked result and
                           answers from it -- no unrelated command after it (no executable tool:
                           the result is canned)
  issue4_warm              the same after an unrelated, finished conversation (a look at /var/log, a
                           log-backup script: the very distraction issue #4 produced), several
                           thousand tokens the server has served once, so the rounds resume from
                           its prefix cache (the cached-token counters at /metrics show how much)
  isolation               concurrent conversations, each with its own planted code, each asked for
                           it back: no answer carries another conversation's code
  stress                   --stress N short requests, --concurrency at a time, alternating session
                           identities: every one completes (200, stop or length)
  limits                   --limits (skipped without): a prompt that nearly fills the served context
                           is answered, one past it is refused naming the context, and a prompt of
                           three quarters of it joins the other slots' decodes, every request ending
  switching                --switch-models A,B (skipped without): requests alternate between two
                           models that do not both fit; each is answered, the load/unload counts
                           show the fleet switched, and the cards' free memory and the pinned host
                           memory (/admin/fleet) do not drift from round to round
  media_image, media_video --image FILE / --video FILE (skipped without): a picture and a clip sent
                           as data: URLs, the answer naming what they show (--image-expect,
                           --video-expect)

  .toolchain\\python\\python.exe -I win\\tools\\qualify_api.py --url http://127.0.0.1:8100 \\
      --model qwen3.8-27b --out out\\qualify-1.jsonl --stress 100 --concurrency 1
"""
import argparse
import http.client
import json
import socket
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

ARGS = None
RESULTS = []
LOG = None
LOCK = threading.Lock()


# ---------------------------------------------------------------- transport

def conn(timeout=300):
    u = urllib.parse.urlparse(ARGS.url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def headers(extra=None):
    h = {"Content-Type": "application/json"}
    if ARGS.key:
        h["Authorization"] = "Bearer " + ARGS.key
    if extra:
        h.update(extra)
    return h


def post(path, body, extra=None, raw=None, timeout=300):
    """One request, whole. Returns (status, parsed JSON or None, text)."""
    c = conn(timeout)
    data = raw if raw is not None else json.dumps(body).encode()
    c.request("POST", path, body=data, headers=headers(extra))
    r = c.getresponse()
    text = r.read().decode("utf-8", "replace")
    c.close()
    try:
        js = json.loads(text)
    except ValueError:
        js = None
    return r.status, js, text


def get(path, timeout=30):
    c = conn(timeout)
    c.request("GET", path, headers=headers())
    r = c.getresponse()
    text = r.read().decode("utf-8", "replace")
    c.close()
    return r.status, text


def sse(path, body, extra=None, stop_after=None, timeout=300):
    """A streamed request. Returns (status, [(event, data-json-or-str)]). With stop_after, the
    connection is closed after that many data events (a client going away)."""
    c = conn(timeout)
    c.request("POST", path, body=json.dumps(body).encode(), headers=headers(extra))
    r = c.getresponse()
    out = []
    if r.status != 200:
        out.append(("error", r.read().decode("utf-8", "replace")))
        c.close()
        return r.status, out
    event = None
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            d = line[5:].strip()
            try:
                d = json.loads(d)
            except ValueError:
                pass
            out.append((event, d))
            event = None
            if stop_after is not None and len(out) >= stop_after:
                # Go away mid-stream, as a client that closed its tab does.
                try:
                    c.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                c.close()
                return r.status, out
    c.close()
    return r.status, out


def healthy():
    try:
        s, _ = get("/health", timeout=10)
        return s == 200
    except OSError:
        return False


# ---------------------------------------------------------------- bookkeeping

def record(name, ok, detail, **extra):
    row = {"check": name, "ok": bool(ok), "detail": detail, "t": time.strftime("%H:%M:%S")}
    row.update(extra)
    with LOCK:
        RESULTS.append(row)
        if LOG:
            LOG.write(json.dumps(row, ensure_ascii=False) + "\n")
            LOG.flush()
    print(("PASS " if ok else "FAIL ") + name + ": " + detail, flush=True)


def check(fn):
    fn.is_check = True
    return fn


def chat_body(messages, **kw):
    b = {"model": ARGS.model, "messages": messages, "max_tokens": kw.pop("max_tokens", 256),
         "temperature": 0}
    b.update(kw)
    return b


def msg_text(js):
    try:
        return js["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def tool_calls(js):
    try:
        return js["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError):
        return []


def fn_tool(name, desc, props, required=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required or []}}}


WEATHER = fn_tool("get_weather", "Current weather for a city",
                  {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, ["city"])


# ---------------------------------------------------------------- the checks

@check
def chat():
    s, js, t = post("/v1/chat/completions",
                    chat_body([{"role": "user", "content": "What is the capital of France? One sentence."}]))
    txt = msg_text(js)
    fin = js and js["choices"][0].get("finish_reason")
    u = js and js.get("usage") or {}
    ok = s == 200 and "Paris" in txt and fin == "stop" and u.get("completion_tokens", 0) > 0
    record("chat", ok, f"status {s}, finish {fin}, {u.get('completion_tokens')} tokens: {txt[:80]!r}")
    return txt


@check
def chat_stream():
    body = chat_body([{"role": "user", "content": "What is the capital of France? One sentence."}])
    _, whole, _ = post("/v1/chat/completions", body)
    body["stream"] = True
    s, ev = sse("/v1/chat/completions", body)
    parts, done, fin = [], False, None
    for _, d in ev:
        if d == "[DONE]":
            done = True
        elif isinstance(d, dict) and d.get("choices"):
            delta = d["choices"][0].get("delta") or {}
            if delta.get("content"):
                parts.append(delta["content"])
            fin = d["choices"][0].get("finish_reason") or fin
    txt = "".join(parts)
    same = txt == msg_text(whole)
    record("chat_stream", s == 200 and done and same and fin == "stop",
           f"status {s}, {len(ev)} events, [DONE] {done}, finish {fin}, equals the whole answer {same}")


@check
def responses():
    s, js, t = post("/v1/responses", {"model": ARGS.model, "input": "What is the capital of France? One sentence.",
                                      "max_output_tokens": 256, "temperature": 0})
    txt = ""
    for item in (js or {}).get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    txt += c.get("text", "")
    st = js and js.get("status")
    record("responses", s == 200 and st == "completed" and "Paris" in txt,
           f"status {s}, response status {st}: {txt[:80]!r}")


@check
def responses_stream():
    s, ev = sse("/v1/responses", {"model": ARGS.model, "input": "What is the capital of France? One sentence.",
                                  "max_output_tokens": 256, "temperature": 0, "stream": True})
    kinds = [(e or (d.get("type") if isinstance(d, dict) else "")) for e, d in ev]
    txt = "".join(d.get("delta", "") for e, d in ev
                  if isinstance(d, dict) and (e or d.get("type")) == "response.output_text.delta")
    ok = s == 200 and "response.completed" in kinds and "Paris" in txt
    record("responses_stream", ok, f"status {s}, {len(ev)} events, completed {'response.completed' in kinds}: {txt[:60]!r}")


@check
def messages():
    s, js, t = post("/v1/messages", {"model": ARGS.model, "max_tokens": 256, "temperature": 0,
                                     "messages": [{"role": "user", "content": "What is the capital of France? One sentence."}]},
                    extra={"anthropic-version": "2023-06-01"})
    txt = "".join(c.get("text", "") for c in (js or {}).get("content", []) if c.get("type") == "text")
    sr = js and js.get("stop_reason")
    record("messages", s == 200 and sr == "end_turn" and "Paris" in txt, f"status {s}, stop {sr}: {txt[:80]!r}")


@check
def messages_stream():
    s, ev = sse("/v1/messages", {"model": ARGS.model, "max_tokens": 256, "temperature": 0, "stream": True,
                                 "messages": [{"role": "user", "content": "What is the capital of France? One sentence."}]},
                extra={"anthropic-version": "2023-06-01"})
    kinds = [(e or (d.get("type") if isinstance(d, dict) else "")) for e, d in ev]
    txt = "".join((d.get("delta") or {}).get("text", "") for e, d in ev
                  if isinstance(d, dict) and (d.get("delta") or {}).get("type") == "text_delta")
    ok = s == 200 and "message_stop" in kinds and "Paris" in txt
    record("messages_stream", ok, f"status {s}, {len(ev)} events, message_stop {'message_stop' in kinds}: {txt[:60]!r}")


@check
def tools_typed():
    tool = fn_tool("book_table", "Book a restaurant table",
                   {"party_size": {"type": "integer"}, "budget_per_person": {"type": "number"},
                    "outdoor": {"type": "boolean"}, "time": {"type": "string", "enum": ["18:00", "19:30", "21:00"]},
                    "dietary": {"type": "array", "items": {"type": "string"}},
                    "contact": {"type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"}},
                                "required": ["name", "phone"]}},
                   ["party_size", "budget_per_person", "outdoor", "time", "dietary", "contact"])
    s, js, _ = post("/v1/chat/completions", chat_body(
        [{"role": "user", "content": "Book a table for 4 people at 19:30, outdoors, 35.5 dollars a person, one vegan "
                                     "and one gluten-free guest. Contact: Dana Lee, 555-0142. Use the tool."}],
        tools=[tool], max_tokens=1024))
    calls = tool_calls(js)
    try:
        a = json.loads(calls[0]["function"]["arguments"])
        ok = (isinstance(a.get("party_size"), int) and not isinstance(a.get("party_size"), bool) and a["party_size"] == 4
              and isinstance(a.get("budget_per_person"), (int, float)) and abs(a["budget_per_person"] - 35.5) < 1e-6
              and a.get("outdoor") is True and a.get("time") == "19:30" and isinstance(a.get("dietary"), list)
              and len(a["dietary"]) == 2 and isinstance(a.get("contact"), dict) and "Dana" in a["contact"].get("name", ""))
        record("tools_typed", ok, f"status {s}: {json.dumps(a)[:200]}")
    except (IndexError, KeyError, ValueError, TypeError) as e:
        record("tools_typed", False, f"status {s}, no parsable call ({e}): {msg_text(js)[:120]!r}")


@check
def tools_multiline():
    tool = fn_tool("write_file", "Write text to a file", {"path": {"type": "string"}, "content": {"type": "string"}},
                   ["path", "content"])
    s, js, _ = post("/v1/chat/completions", chat_body(
        [{"role": "user", "content": "Write a three-line haiku about rain to the file haiku.txt, one line per line. "
                                     "Use the tool."}], tools=[tool], max_tokens=1024))
    calls = tool_calls(js)
    try:
        a = json.loads(calls[0]["function"]["arguments"])
        lines = [x for x in a.get("content", "").split("\n") if x.strip()]
        record("tools_multiline", len(lines) >= 3, f"status {s}, {len(lines)} lines in content: {a.get('content', '')[:120]!r}")
    except (IndexError, KeyError, ValueError, TypeError) as e:
        record("tools_multiline", False, f"status {s}, no parsable call ({e})")


@check
def tools_empty():
    tool = fn_tool("get_server_time", "The server's current time. Takes no arguments.", {})
    s, js, _ = post("/v1/chat/completions", chat_body(
        [{"role": "user", "content": "What time is it on the server? Use the tool."}], tools=[tool], max_tokens=1024))
    calls = tool_calls(js)
    try:
        raw = calls[0]["function"].get("arguments") or "{}"
        a = json.loads(raw)
        record("tools_empty", calls[0]["function"]["name"] == "get_server_time" and a == {},
               f"status {s}, arguments {raw!r}")
    except (IndexError, KeyError, ValueError, TypeError) as e:
        record("tools_empty", False, f"status {s}, no parsable call ({e})")


@check
def tools_multiple():
    msgs = [{"role": "user", "content": "What is the weather in Paris and in Tokyo right now? Use the tool for each city."}]
    cities = set()
    for _ in range(3):
        s, js, _ = post("/v1/chat/completions", chat_body(msgs, tools=[WEATHER], max_tokens=1024))
        calls = tool_calls(js)
        if not calls:
            break
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        for c in calls:
            try:
                city = json.loads(c["function"]["arguments"]).get("city", "")
            except ValueError:
                city = ""
            cities.add(city.lower())
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                         "content": json.dumps({"city": city, "temp_c": 14 if "par" in city.lower() else 22})})
        if {"paris", "tokyo"} <= {x.split(",")[0].strip() for x in cities}:
            break
    got = {x.split(",")[0].strip() for x in cities}
    record("tools_multiple", {"paris", "tokyo"} <= got, f"cities called: {sorted(got)}")


def _roundtrip_ok(txt):
    return "14" in txt and ("rain" in txt.lower())


@check
def tool_roundtrip_chat():
    msgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
    s, js, _ = post("/v1/chat/completions", chat_body(msgs, tools=[WEATHER], max_tokens=1024))
    calls = tool_calls(js)
    if not calls:
        record("tool_roundtrip_chat", False, f"status {s}, no call")
        return
    msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
    msgs.append({"role": "tool", "tool_call_id": calls[0].get("id", ""),
                 "content": '{"city":"Paris","temp_c":14,"sky":"light rain"}'})
    s, js, _ = post("/v1/chat/completions", chat_body(msgs, tools=[WEATHER], max_tokens=1024))
    txt = msg_text(js)
    record("tool_roundtrip_chat", s == 200 and _roundtrip_ok(txt), f"status {s}: {txt[:100]!r}")


@check
def tool_roundtrip_responses():
    tool = {"type": "function", "name": "get_weather", "description": "Current weather for a city",
            "parameters": WEATHER["function"]["parameters"]}
    inp = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
    s, js, _ = post("/v1/responses", {"model": ARGS.model, "input": inp, "tools": [tool],
                                      "max_output_tokens": 1024, "temperature": 0})
    fc = [o for o in (js or {}).get("output", []) if o.get("type") == "function_call"]
    if not fc:
        record("tool_roundtrip_responses", False, f"status {s}, no function_call")
        return
    inp += [fc[0], {"type": "function_call_output", "call_id": fc[0].get("call_id", ""),
                    "output": '{"city":"Paris","temp_c":14,"sky":"light rain"}'}]
    s, js, _ = post("/v1/responses", {"model": ARGS.model, "input": inp, "tools": [tool],
                                      "max_output_tokens": 1024, "temperature": 0})
    txt = "".join(c.get("text", "") for o in (js or {}).get("output", []) if o.get("type") == "message"
                  for c in o.get("content", []))
    record("tool_roundtrip_responses", s == 200 and _roundtrip_ok(txt), f"status {s}: {txt[:100]!r}")


@check
def tool_roundtrip_messages():
    tool = {"name": "get_weather", "description": "Current weather for a city",
            "input_schema": WEATHER["function"]["parameters"]}
    msgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
    ex = {"anthropic-version": "2023-06-01"}
    s, js, _ = post("/v1/messages", {"model": ARGS.model, "max_tokens": 1024, "temperature": 0,
                                     "messages": msgs, "tools": [tool]}, extra=ex)
    use = [c for c in (js or {}).get("content", []) if c.get("type") == "tool_use"]
    if not use:
        record("tool_roundtrip_messages", False, f"status {s}, no tool_use (stop {js and js.get('stop_reason')})")
        return
    msgs.append({"role": "assistant", "content": js["content"]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": use[0]["id"],
                                              "content": '{"city":"Paris","temp_c":14,"sky":"light rain"}'}]})
    s, js, _ = post("/v1/messages", {"model": ARGS.model, "max_tokens": 1024, "temperature": 0,
                                     "messages": msgs, "tools": [tool]}, extra=ex)
    txt = "".join(c.get("text", "") for c in (js or {}).get("content", []) if c.get("type") == "text")
    record("tool_roundtrip_messages", s == 200 and _roundtrip_ok(txt), f"status {s}: {txt[:100]!r}")


def _after_cancel(name, detail):
    time.sleep(1.0)
    ok = healthy()
    s, js, _ = post("/v1/chat/completions", chat_body([{"role": "user", "content": "Say OK."}], max_tokens=64))
    record(name, ok and s == 200 and js is not None, f"{detail}; healthy after {ok}, next request {s}")


@check
def cancel_prefill():
    # A long prompt, gone before its first token: the stream is closed after the first event.
    long = ("The quick brown fox jumps over the lazy dog. " * 400)
    s, ev = sse("/v1/chat/completions", chat_body([{"role": "user", "content": long + "\nSummarise."}],
                                                    max_tokens=512, stream=True), stop_after=1)
    _after_cancel("cancel_prefill", f"closed after {len(ev)} event(s)")


@check
def cancel_decode():
    s, ev = sse("/v1/chat/completions", chat_body([{"role": "user", "content": "Count from 1 to 500."}],
                                                    max_tokens=2048, stream=True), stop_after=12)
    _after_cancel("cancel_decode", f"closed after {len(ev)} event(s)")


@check
def malformed():
    cases = [
        ("bad json", None, b'{"model": "x", "messages": [', "/v1/chat/completions"),
        ("unknown model", {"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]}, None,
         "/v1/chat/completions"),
        ("no messages", {"model": ARGS.model}, None, "/v1/chat/completions"),
        ("messages not a list", {"model": ARGS.model, "messages": "hi"}, None, "/v1/chat/completions"),
        ("max_tokens a string", {"model": ARGS.model, "messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": "many"}, None, "/v1/chat/completions"),
        ("bad role", {"model": ARGS.model, "messages": [{"role": "wizard", "content": "hi"}]}, None,
         "/v1/chat/completions"),
        ("responses without input", {"model": ARGS.model}, None, "/v1/responses"),
    ]
    # Accepted by upstream on purpose (the same on Linux): a non-positive max_tokens and a Messages
    # request without max_tokens take the server's default. They must answer, not fail or hang.
    lenient = [
        ("negative max_tokens", {"model": ARGS.model, "messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": -5}, None, "/v1/chat/completions"),
        ("messages without max_tokens", {"model": ARGS.model, "messages": [{"role": "user", "content": "hi"}]}, None,
         "/v1/messages"),
    ]
    bad, notes = [], []
    for what, body, raw, path in cases:
        try:
            s, js, t = post(path, body, raw=raw, timeout=60)
        except OSError as e:
            bad.append(f"{what}: {e}")
            continue
        if not (400 <= s < 500) or js is None or "error" not in js:
            bad.append(f"{what}: {s} {t[:80]!r}")
    for what, body, raw, path in lenient:
        try:
            s, js, t = post(path, body, raw=raw, timeout=300)
        except OSError as e:
            bad.append(f"{what}: {e}")
            continue
        if s >= 500 or js is None:
            bad.append(f"{what}: {s} {t[:80]!r}")
        else:
            notes.append(f"{what}: {s}")
    ok_after = healthy()
    record("malformed", not bad and ok_after,
           f"{len(cases)} refused with a 4xx and an error body; accepted as upstream does: {notes}; healthy after {ok_after}"
           + (f"; wrong: {bad}" if bad else ""))


ISSUE4_SYSTEM = ("You are an assistant with access to the user's computer through the run_shell tool. Use it when "
                 "you need information from the system. Answer the user's question and nothing else.")
SHELL = fn_tool("run_shell", "Run a shell command on the user's machine and return its output",
                {"command": {"type": "string", "description": "The command line to run"}}, ["command"])
DF_OUT = ("Filesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  915G  402G  467G  47% /\n"
          "tmpfs            32G  1.1M   32G   1% /run\n/dev/nvme1n1p1  3.6T  1.2T  2.4T  34% /data\n")
DISK_WORDS = ("df", "du ", "disk", "free", "lsblk", "Get-PSDrive", "Get-Volume", "fsutil", "wmic logicaldisk", "statvfs")


OS_WORDS = ("uname", "os-release", "lsb_release", "hostnamectl", "systeminfo", "ver", "ostype", "osversion",
            "sw_vers", "whoami")
UNAME_OUT = "Linux workstation 6.8.0-48-generic #48-Ubuntu SMP PREEMPT_DYNAMIC x86_64 x86_64 x86_64 GNU/Linux\n"


# Unrelated earlier turns for issue4_warm: a look at /var/log through the same tool, then the
# log-backup script issue #4's model ran instead of answering, so the history holds exactly the
# distraction the bug produced. The listing makes it several thousand tokens, past the 2048-token
# chunk at which an exact prefix resume can start, so the rounds can resume from the cache. A line is
# about 33 tokens: 240 lines (some 8.4K tokens with the rest) where the context allows, fewer on a
# smaller one, never under 70 (2.3K).
def log_listing(lines):
    return "total 48212\n" + "".join(
        f"-rw-r----- 1 syslog adm {4096 + 977 * i:>8} Nov {1 + i % 28:>2} 0{i % 10}:{(7 * i) % 60:02d} "
        f"{('syslog', 'kern.log', 'auth.log', 'dpkg.log', 'ufw.log', 'boot.log')[i % 6]}.{i // 6 + 1}.gz\n"
        for i in range(lines))


def listing_lines(max_model_len):
    """The most listing lines that leave half the context for the rounds."""
    if not max_model_len:
        return 240
    return min(240, max(70, (max_model_len // 2 - 1000) // 33))


def warm_history(lines=240):
    return [
    {"role": "user", "content": "What is in /var/log on this machine?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_warm_1", "type": "function",
         "function": {"name": "run_shell", "arguments": json.dumps({"command": "ls -l /var/log"})}}]},
    {"role": "tool", "tool_call_id": "call_warm_1", "content": log_listing(lines)},
    {"role": "assistant", "content": "/var/log holds rotated, gzip-compressed archives of syslog, kern.log, "
                                     f"auth.log, dpkg.log, ufw.log and boot.log, {lines} files in all."},
    {"role": "user", "content": "Can you write me a bash script that backs up /var/log into /backup every night "
                                "and keeps the last 7 archives?"},
    {"role": "assistant", "content": "Here is one:\n\n```bash\n#!/usr/bin/env bash\nset -euo pipefail\n"
                                     "dest=/backup\nstamp=$(date +%F)\nmkdir -p \"$dest\"\n"
                                     "tar -czf \"$dest/logs-$stamp.tar.gz\" -C / var/log\n"
                                     "ls -1t \"$dest\"/logs-*.tar.gz | tail -n +8 | xargs -r rm --\n```\n\n"
                                     "Run it from cron, for example `0 2 * * * /usr/local/bin/backup-logs.sh`."},
    {"role": "user", "content": "Thanks. What does the -z flag of tar do?"},
    {"role": "assistant", "content": "`-z` filters the archive through gzip, so the backup is compressed "
                                     "(`.tar.gz`). Without it tar writes an uncompressed archive."},
    ]


def served_max_model_len():
    """The --model's max_model_len from /v1/models, or 0."""
    s, text = get("/v1/models", timeout=30)
    if s != 200:
        return 0
    for m in json.loads(text).get("data", []):
        if m.get("id") == ARGS.model:
            return int(m.get("max_model_len") or 0)
    return 0


@check
def issue4():
    """The model may first ask what system it is on (uname and the like) -- reasonable, answered with a
    canned Linux. What issue #4 reported is the turn after the disk-space result: an unrelated script
    (a log backup) instead of the answer. So every command must be about the system or its disks, at
    most four of them, and the conversation must end in an answer that uses the canned figures."""
    issue4_rounds("issue4", [])


@check
def issue4_warm():
    """issue4 after unrelated history (fresh and unrelated warm history): the same
    question asked after a finished conversation about the logs and a log-backup script, whose
    prefix the server has served once (a first request over it), so the rounds can resume from the
    cache. The server's cached-token counters (/metrics) say how much was resumed. The history is
    sized to the served context (listing_lines)."""
    lines = listing_lines(served_max_model_len())
    history = warm_history(lines)
    s, js, _ = post("/v1/chat/completions", chat_body(
        [{"role": "system", "content": ISSUE4_SYSTEM}] + history[:-1], tools=[SHELL], max_tokens=256))
    if s != 200:
        record("issue4_warm", False, f"the warming request ({lines} listing lines): status {s}")
        return
    before = cached_counters()
    issue4_rounds("issue4_warm", history,
                  lambda: f"; {lines} listing lines; cached tokens over the rounds " + ", ".join(
                      f"{k.split(':')[1]} +{cached_counters().get(k, 0) - v:.0f}" for k, v in before.items()))


def cached_counters():
    """The server's prefix-cache counters (Prometheus text at /metrics); {} when it has none."""
    try:
        s, text = get("/metrics", timeout=10)
    except OSError:
        return {}
    out = {}
    for line in text.splitlines() if s == 200 else []:
        for k in ("radiance:attn_cached_tokens_total", "radiance:linear_cached_tokens_total"):
            if line.startswith(k):
                out[k] = out.get(k, 0) + float(line.rsplit(" ", 1)[1])
    return out


def issue4_rounds(name, history, extra=lambda: ""):
    rounds = ARGS.issue4_rounds
    fails, paths = [], []
    for k in range(rounds):
        msgs = [{"role": "system", "content": ISSUE4_SYSTEM}] + list(history) + \
               [{"role": "user", "content": "How much free disk space does the system have?"}]
        temp = 0 if k == 0 else 0.7
        cmds, answer, why = [], None, None
        for step in range(5):
            s, js, _ = post("/v1/chat/completions", chat_body(msgs, tools=[SHELL], max_tokens=2048,
                                                              temperature=temp, seed=1000 * (step + 1) + k))
            calls = tool_calls(js)
            if not calls:
                answer = msg_text(js)
                break
            if step == 4:
                why = f"still calling tools after {len(cmds)} commands"
                break
            msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
            for c in calls:
                try:
                    cmd = json.loads(c["function"]["arguments"] or "{}").get("command", "")
                except ValueError:
                    cmd = c["function"].get("arguments", "")
                cmds.append(cmd)
                low = cmd.lower()
                if any(w.lower() in low for w in DISK_WORDS):
                    out = DF_OUT
                elif any(w in low for w in OS_WORDS):
                    out = UNAME_OUT
                else:
                    why = f"an unrelated command: {cmd[:100]!r}"
                    break
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
            if why:
                break
        paths.append(cmds)
        if why:
            fails.append(f"round {k}: {why} (commands {cmds})")
        elif answer is None:
            fails.append(f"round {k}: no answer")
        elif cmds and not any(x in answer for x in ("467", "2.4T", "2.4 T", "2.4 TB")):
            fails.append(f"round {k}: the answer does not use the result: {answer[:100]!r}")
    record(name, not fails, f"{rounds} round(s), commands per round {paths}" +
           (f"; {fails}" if fails else "; every one on topic and answered from the result") + extra())


@check
def isolation():
    n = max(2, ARGS.concurrency * 2)
    codes = [f"{1000 + 37 * i:04d}-{chr(65 + i % 26)}" for i in range(n)]

    def one(i):
        msgs = [{"role": "system", "content": f"Session {i}."},
                {"role": "user", "content": f"Remember this access code exactly: {codes[i]}. Reply only 'noted'."},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "What is the access code I gave you? Reply with the code only."}]
        s, js, _ = post("/v1/chat/completions", chat_body(msgs, max_tokens=512, user=f"session-{i}"))
        return i, s, msg_text(js)

    with ThreadPoolExecutor(max_workers=max(1, ARGS.concurrency)) as ex:
        out = list(ex.map(one, range(n)))
    wrong = []
    for i, s, txt in out:
        others = [c for j, c in enumerate(codes) if j != i and c in txt]
        if s != 200 or codes[i] not in txt or others:
            wrong.append((i, s, txt[:60], others))
    record("isolation", not wrong, f"{n} conversations, {ARGS.concurrency} at a time" + (f"; wrong: {wrong}" if wrong else ""))


@check
def stress():
    n = ARGS.stress
    if n <= 0:
        return
    prompts = ["Name a prime number between 10 and 20.", "Give one synonym for 'happy'.",
               "What is 12 times 12?", "Name a planet in the solar system.", "Translate 'cat' to French."]

    def one(i):
        t0 = time.time()
        try:
            s, js, _ = post("/v1/chat/completions", chat_body(
                [{"role": "system", "content": f"You are assistant #{i % 7}."},
                 {"role": "user", "content": prompts[i % len(prompts)]}],
                max_tokens=256, temperature=0 if i % 2 else 0.8, user=f"user-{i % 5}"), timeout=600)
            fin = js and js["choices"][0].get("finish_reason")
            return s, fin, time.time() - t0
        except OSError as e:
            return str(e), None, time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, ARGS.concurrency)) as ex:
        out = list(ex.map(one, range(n)))
    wall = time.time() - t0
    bad = [(i, s, f) for i, (s, f, _) in enumerate(out) if s != 200 or f not in ("stop", "length")]
    lat = sorted(x[2] for x in out)
    record("stress", not bad and healthy(),
           f"{n} requests, {ARGS.concurrency} at a time, {wall:.0f} s; latency median {lat[len(lat) // 2]:.2f} s, "
           f"p95 {lat[int(len(lat) * 0.95) - 1]:.2f} s" + (f"; failed: {bad[:10]}" if bad else ""))


# ---------------------------------------------------------------- media

def data_url(path, mime):
    import base64
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def media_answer(part, question):
    """A chat request with one media part and a question; (status, answer text)."""
    msgs = [{"role": "user", "content": [part, {"type": "text", "text": question}]}]
    s, js, _ = post("/v1/chat/completions", chat_body(msgs, max_tokens=1024), timeout=600)
    return s, msg_text(js)


@check
def media_image():
    """--image FILE: a picture the model must describe. The default file a qualification uses is a
    green square on white (win/tools/package-check's media folder makes one with ffmpeg); the answer
    must name its colour (--image-expect, default 'green')."""
    if not ARGS.image:
        return
    mime = "image/png" if ARGS.image.lower().endswith(".png") else "image/jpeg"
    s, txt = media_answer({"type": "image_url", "image_url": {"url": data_url(ARGS.image, mime)}},
                          "What shape and colour is in the middle of this picture? Answer in one sentence.")
    ok = s == 200 and all(w in txt.lower() for w in ARGS.image_expect.split(","))
    record("media_image", ok, f"status {s}: {txt.strip()[:120]!r}")


@check
def media_video():
    """--video FILE: a clip the model must describe, decoded through libavformat (the default a
    qualification uses is 1.5 s of red, then 1.5 s of blue, H.264); the answer must name each colour
    in --video-expect (default 'red,blue')."""
    if not ARGS.video:
        return
    s, txt = media_answer({"type": "video_url", "video_url": {"url": data_url(ARGS.video, "video/mp4")}},
                          "What colours does this video show, in order? Answer in one sentence.")
    ok = s == 200 and all(w in txt.lower() for w in ARGS.video_expect.split(","))
    record("media_video", ok, f"status {s}: {txt.strip()[:120]!r}")


# ---------------------------------------------------------------- limits

LIMIT_UNIT = "The lighthouse keeper wrote down every ship that passed the rocks before dawn. "


def prompt_of(tokens, unit_tokens):
    """About `tokens` tokens of LIMIT_UNIT, and its count of units."""
    k = max(tokens // max(unit_tokens, 1), 1)
    return LIMIT_UNIT * k, k


@check
def limits():
    """--limits: the served model's advertised context and concurrency (max_model_len from /v1/models,
    the slots from /props). A prompt that nearly fills the context is answered; one past it is refused
    with a 4xx that names the context; and a prompt of three quarters of the context, sent while every
    other slot decodes, joins them: each request finishes with stop or length, none is refused."""
    if not ARGS.limits:
        return
    L = served_max_model_len()
    s, text = get("/props", timeout=30)
    slots = int((json.loads(text) if s == 200 else {}).get("total_slots") or 1)
    s, js, _ = post("/tokenize", {"model": ARGS.model, "content": LIMIT_UNIT * 10})
    unit = len((js or {}).get("tokens", [])) // 10
    if not L or not unit:
        record("limits", False, f"no max_model_len ({L}) or no /tokenize ({s})")
        return
    ask = "\n\nHow many times is the sentence above repeated, roughly? Answer in one short sentence."
    parts = []

    near, k = prompt_of(L - 700, unit)
    t0 = time.time()
    s1, js1, _ = post("/v1/chat/completions", chat_body([{"role": "user", "content": near + ask}],
                      max_tokens=64, chat_template_kwargs={"enable_thinking": False}), timeout=900)
    u1 = (js1 or {}).get("usage", {})
    ok1 = s1 == 200 and bool(msg_text(js1))
    parts.append(f"near-full ({k} units, {u1.get('prompt_tokens')} of {L} tokens) -> {s1} in "
                 f"{time.time() - t0:.1f} s: {msg_text(js1).strip()[:60]!r}")

    over, k = prompt_of(L + 600, unit)
    s2, js2, _ = post("/v1/chat/completions", chat_body([{"role": "user", "content": over + ask}],
                      max_tokens=16), timeout=300)
    err = ((js2 or {}).get("error") or {}).get("message", "")
    ok2 = 400 <= s2 < 500 and "context" in err
    parts.append(f"past the context ({k} units) -> {s2}: {err[:90]!r}")

    results = []

    def decode(i):
        st, j, _ = post("/v1/chat/completions", chat_body(
            [{"role": "user", "content": f"Write a long story about lighthouse number {i}."}],
            max_tokens=400, chat_template_kwargs={"enable_thinking": False}), timeout=900)
        results.append(("decode", st, ((j or {}).get("choices") or [{}])[0].get("finish_reason")))

    threads = [threading.Thread(target=decode, args=(i,)) for i in range(max(slots - 1, 0))]
    for th in threads:
        th.start()
    time.sleep(2.0)
    long_p, k = prompt_of(L * 3 // 4, unit)
    # Its own opening, so the near-full prompt's cached prefix does not serve it: a real prefill.
    long_p = f"Logbook {int(time.time() * 1000) % 1000003}, a different night.\n" + long_p
    t0 = time.time()
    s3, js3, _ = post("/v1/chat/completions", chat_body([{"role": "user", "content": long_p + ask}],
                      max_tokens=64, chat_template_kwargs={"enable_thinking": False}), timeout=900)
    results.append(("long", s3, ((js3 or {}).get("choices") or [{}])[0].get("finish_reason")))
    dt = time.time() - t0
    cached = ((js3 or {}).get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    for th in threads:
        th.join()
    ok3 = all(st == 200 and fin in ("stop", "length") for _, st, fin in results)
    parts.append(f"a {k}-unit prompt ({cached} tokens from the cache) joining {len(threads)} decoding "
                 f"stream(s) -> {s3} in {dt:.1f} s; "
                 f"every request {[(w, st, fin) for w, st, fin in results]}")
    record("limits", ok1 and ok2 and ok3 and healthy(), f"{slots} slot(s), context {L}; " + "; ".join(parts))


# ---------------------------------------------------------------- switching

def model_states():
    """{model id: (state, arrivals, departures)} from /v1/models. An arrival is a load or a restore
    from a park, a departure an unload or a park: with virtual memory the fleet parks a model that
    makes room and restores it, without it the model unloads and loads again."""
    s, text = get("/v1/models", timeout=30)
    if s != 200:
        return {}
    out = {}
    for m in json.loads(text).get("data", []):
        out[m["id"]] = (m.get("state"), m.get("loads", 0) + m.get("restores", 0),
                        m.get("unloads", 0) + m.get("parks", 0), m.get("parks", 0))
    return out


def fleet_memory():
    """(the cards' free bytes summed, the bytes the process pins) from /admin/fleet, or None."""
    s, text = get("/admin/fleet", timeout=30)
    if s != 200:
        return None
    j = json.loads(text)
    return (sum(c.get("free_bytes", 0) for c in j.get("cards", [])),
            j.get("host", {}).get("held_bytes", 0))


@check
def switching():
    """--switch-models A,B --switch-rounds N: a fleet of two models that do not both fit:
    requests alternate between them, each answered (the fleet parks or unloads one and restores or
    loads the other as a request names it), and the models' counts show the switches happened.

    After every round the same model is resident, so /admin/fleet's figures (the cards' free memory,
    the host memory the process pins) are read then: from the end of the first round, which is after
    each model has loaded once, to the last, neither may drift by more than --switch-drift-mib."""
    if not ARGS.switch_models:
        return
    names = ARGS.switch_models.split(",")
    before = model_states()
    bad, times, mem = [], [], []
    for k in range(ARGS.switch_rounds * len(names)):
        m = names[k % len(names)]
        t0 = time.time()
        body = chat_body([{"role": "user", "content": "Name one primary colour. One word."}], max_tokens=512)
        body["model"] = m
        s, js, _ = post("/v1/chat/completions", body, timeout=900)
        dt = time.time() - t0
        times.append(dt)
        txt = msg_text(js) or (js and js["choices"][0]["message"].get("reasoning_content")) or ""
        if s != 200 or not txt:
            bad.append((k, m, s))
        if k % len(names) == len(names) - 1:
            mem.append(fleet_memory())
    after = model_states()
    none = (None, 0, 0, 0)
    switches = {m: (after.get(m, none)[1] - before.get(m, none)[1],
                    after.get(m, none)[2] - before.get(m, none)[2]) for m in names}
    parks = sum(after.get(m, none)[3] - before.get(m, none)[3] for m in names)
    # Every request after the first names the model that is not resident, so each one brings it in.
    # Fewer arrivals mean both stayed resident and nothing was switched (a fleet where both fit, or
    # a group rule that let both in): answered, but not the check.
    loads = sum(n for n, _ in switches.values())
    switched = loads >= len(times) - 1
    # Memory after each round, the first round's as the baseline: a card's free memory that shrinks,
    # or pinned host memory that grows, from switch to switch is a leak.
    drift_ok, drift = True, ""
    got = [x for x in mem if x]
    if len(got) >= 2:
        mib = 1 << 20
        free_lost = (got[0][0] - min(f for f, _ in got)) / mib
        held_grew = (max(h for _, h in got) - got[0][1]) / mib
        drift_ok = free_lost <= ARGS.switch_drift_mib and held_grew <= ARGS.switch_drift_mib
        drift = (f"; after {len(got)} rounds the cards' free memory fell by at most {free_lost:.0f} MiB "
                 f"and the pinned host memory rose by at most {held_grew:.0f} MiB "
                 f"(free {got[0][0] / mib:.0f} -> {got[-1][0] / mib:.0f} MiB, "
                 f"pinned {got[0][1] / mib:.0f} -> {got[-1][1] / mib:.0f} MiB)")
    elif ARGS.switch_rounds >= 2:
        drift = "; no memory figures (/admin/fleet did not answer)"
    record("switching", not bad and switched and drift_ok and healthy(),
           f"{len(times)} requests alternating {names}; arrivals/departures during the run {switches}, "
           f"{parks} of the departures parks; "
           f"slowest {max(times):.1f} s, median {sorted(times)[len(times) // 2]:.1f} s" + drift +
           (f"; failed: {bad}" if bad else "") +
           ("" if switched else f"; only {loads} load(s) or restore(s) for {len(times)} requests: the models did not take turns"))


# ---------------------------------------------------------------- main

def main():
    global ARGS, LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-file", help="a file holding the key (kept off the command line)")
    ap.add_argument("--out", help="JSON lines, one per check")
    ap.add_argument("--only", help="comma-separated check names")
    ap.add_argument("--skip", help="comma-separated check names")
    ap.add_argument("--stress", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--issue4-rounds", type=int, default=5)
    ap.add_argument("--image", help="a picture for media_image (skipped without one)")
    ap.add_argument("--image-expect", default="green", help="comma-separated words the answer must hold")
    ap.add_argument("--video", help="an MP4 for media_video (skipped without one)")
    ap.add_argument("--video-expect", default="red,blue", help="comma-separated words the answer must hold")
    ap.add_argument("--limits", action="store_true",
                    help="run the limits check: a near-full context, one past it, a long prompt joining decodes")
    ap.add_argument("--switch-models", help="A,B: two models of the fleet for the switching check")
    ap.add_argument("--switch-rounds", type=int, default=3)
    ap.add_argument("--switch-drift-mib", type=int, default=256,
                    help="the most the cards' free memory may fall, or pinned host memory rise, over the rounds")
    ARGS = ap.parse_args()
    # Answers carry emoji and other text a Windows console code page cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ARGS.key = open(ARGS.api_key_file).read().strip() if ARGS.api_key_file else None
    if ARGS.out:
        LOG = open(ARGS.out, "a", encoding="utf-8")
    if not healthy():
        print("the server does not answer /health", file=sys.stderr)
        return 2
    names = [n for n, f in globals().items() if callable(f) and getattr(f, "is_check", False)]
    if ARGS.only:
        names = [n for n in names if n in ARGS.only.split(",")]
    if ARGS.skip:
        names = [n for n in names if n not in ARGS.skip.split(",")]
    for n in names:
        try:
            globals()[n]()
        except Exception as e:  # a check that throws is a failed check, with why
            record(n, False, f"{type(e).__name__}: {e}")
    failed = [r["check"] for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
