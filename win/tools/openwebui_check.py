"""Brilliance: tool round trips through Open WebUI against a served model, standard library only.

  .toolchain\\python\\python.exe -I win\\tools\\openwebui_check.py --webui http://127.0.0.1:8180 \\
      --model flashnext --out out\\clients\\runs\\openwebui-1 [--repeat 10] [--no-session]

Open WebUI (0.13.3 here) runs with WEBUI_AUTH=False and one OpenAI connection to the server; see
win/tools/openwebui-start.ps1. The route is the one its own frontend takes (read from its source, and
its built page's request):
  1. POST /api/v1/auths/signin (any credentials with auth off) for a token;
  2. POST /api/v1/tools/create: a Python `Tools` class with two controlled tools, disk_free (a fixed
     "123.4 GB free") and read_note (HELIOTROPE for the note "secret"), executed inside Open WebUI;
  3. per turn, POST /api/chat/completions with stream true, tool_ids, native function calling, a fresh
     id for the reply, and the turn as the frontend states it: `user_message` (its own id, parentId
     and content), `parent_id`, `session_id` and `background_tasks`. No `messages`: Open WebUI stores
     the user message and builds the model's messages from the chat it keeps, earlier tool calls and
     their results included. A first turn names no chat: Open WebUI makes one and answers with its
     chat_id. A later turn names the chat and, as parent, the reply before it.
     Open WebUI runs the whole tool loop against the server (the model's tool call, the tool, the
     result back) as a task of its own and answers the POST at once;
  4. GET /api/v1/chats/<id> until the stored reply is done: its text, its function_call items and
     the function_call_output items the tools produced;
  5. at the end, GET /api/tasks/chat/<id> until every chat's tasks are over.

WHAT session_id CHANGES. With one (the browser always has one: its socket's), Open WebUI adds its
built-in tools to the two here (32 tools and a prompt of about 6,200 tokens in 0.13.3) and runs the
turn as a background task; the browser's background tasks (a title, tags and follow-up questions,
each a request of its own to the server) run too. The id here is made up: no socket is connected,
so the events Open WebUI emits go nowhere and everything is read from the stored chat.
--no-session leaves it out: only the two tools are offered (a prompt of about 400 tokens), no
background tasks run, and the chat is made first with POST /api/v1/chats/new.

The cases, each in a chat of its own unless it continues one:
  disk-N      "How much free disk space ..." (issue #4's question): disk_free is called and nothing
              else, the tool's own result is stored, and the answer gives 123.4;
  followup-N  disk-N's chat continued: the figure asked for again. It must come from the chat's
              history (no tool call this turn), so the server saw the earlier call and its result;
  note-N      the secret word read with read_note("secret"): HELIOTROPE in the answer;
  isolation   a new chat asked for a secret word: no tool call, and HELIOTROPE nowhere in the
              stored reply (text, reasoning or items);
  cache, cache-followup
              a chat that opens with a long inventory (this run's own) and a question about it,
              then asks for the code in it: the code comes back, and the server reused more of the
              second turn's prompt (the stored reply's usage.cache_n) than a chat can have in common
              with another, which only this chat's own first turn can explain.
Each stored reply goes to --out\\<case>.json and the verdicts to --out\\summary.json.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid

TOOLS_SRC = '''class Tools:
    def __init__(self):
        pass

    def disk_free(self) -> str:
        """
        Report the free disk space on this system's main drive.
        """
        return "Drive C: 123.4 GB free of 931.5 GB"

    def read_note(self, name: str) -> str:
        """
        Read a saved note by its name.
        :param name: The note's name, for example "secret".
        """
        if name.strip().lower().startswith("secret"):
            return "The secret word is HELIOTROPE."
        return "No note named " + name + "."
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--webui", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600, help="seconds a reply may take")
    ap.add_argument("--no-session", action="store_true",
                    help="no session_id: only the two test tools are offered, and no background tasks run")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    token = None
    session = None if a.no_session else str(uuid.uuid4())

    def call(method, path, body=None, timeout=60):
        req = urllib.request.Request(a.webui + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:   # Open WebUI's reason is in the body
            e.msg = f"{e.msg}: {e.read()[:300].decode('utf-8', 'replace')}"
            raise
        return json.loads(raw) if raw.strip() else None

    token = call("POST", "/api/v1/auths/signin", {"email": "x", "password": "x"})["token"]
    tool_id = "brill_test"
    try:
        call("POST", "/api/v1/tools/create", {"id": tool_id, "name": "Brilliance test tools",
                                              "meta": {"description": "controlled tools"},
                                              "content": TOOLS_SRC})
    except urllib.error.HTTPError as e:
        if e.code != 400:   # 400: it exists from an earlier run
            raise
    results, ok, chats = {}, True, []

    def ask(name, prompt, after=None):
        """One turn, in a new chat or (after: an earlier turn's result) continuing that one."""
        parent = after["id"] if after else None
        chat_id = after["chat"] if after else None
        if chat_id is None and a.no_session:   # without a session Open WebUI does not say which chat it made
            chat_id = call("POST", "/api/v1/chats/new", {"chat": {"title": name, "models": [a.model]}})["id"]
        mid = str(uuid.uuid4())
        user = {"id": str(uuid.uuid4()), "parentId": parent, "childrenIds": [], "role": "user", "content": prompt,
                "timestamp": int(time.time()), "models": [a.model]}
        body = {"stream": True, "model": a.model, "params": {"function_calling": "native"}, "tool_ids": [tool_id],
                "features": {}, "id": mid, "parent_id": parent, "user_message": user}
        if chat_id:
            body["chat_id"] = chat_id
        if session:
            # The frontend's defaults: a title and tags after a chat's first turn, follow-ups after every turn.
            body["session_id"] = session
            body["background_tasks"] = dict({"follow_up_generation": True},
                                            **({} if parent else {"title_generation": True, "tags_generation": True}))
        t0 = time.time()
        err, msg, turns = None, {}, 0
        try:
            reply = call("POST", "/api/chat/completions", body, timeout=a.timeout) or {}
            err = reply.get("error")
            chat_id = chat_id or reply.get("chat_id")
        except Exception as e:   # noqa: BLE001  (recorded with the case)
            err = repr(e)
        if chat_id and chat_id not in chats:
            chats.append(chat_id)
        while chat_id and not err:
            history = call("GET", f"/api/v1/chats/{chat_id}")["chat"].get("history", {}).get("messages", {})
            msg, turns = history.get(mid, {}), len(history)
            if msg.get("done"):
                break
            if time.time() - t0 > a.timeout:
                err = f"no reply in {a.timeout} s"
                break
            time.sleep(0.25)
        secs = time.time() - t0
        (out / f"{name}.json").write_text(json.dumps(msg, indent=1, ensure_ascii=False), encoding="utf-8")
        items = msg.get("output") or []
        calls = [o.get("name") for o in items if o.get("type") == "function_call"]
        made = [" ".join(p.get("text", "") for p in o.get("output") or [] if isinstance(p, dict))
                for o in items if o.get("type") == "function_call_output"]
        usage = msg.get("usage") or {}   # the server's, for the reply's last request: cache_n of the prompt reused
        return dict(secs=secs, err=err or msg.get("error"), done=msg.get("done"), tools=calls, results=made,
                    text=(msg.get("content") or "").strip(), chat=chat_id, id=mid, stored=turns,
                    cached=usage.get("cache_n"), prompt=usage.get("input_tokens"),
                    all=json.dumps(msg, ensure_ascii=False))

    def check(name, r, passed, why):
        nonlocal ok
        ok &= passed
        results[name] = dict({k: v for k, v in r.items() if k != "all"}, passed=passed)
        print(f"{'PASS' if passed else 'FAIL'} {name}: {why}; {r['secs']:.1f} s, done {r['done']}, "
              f"error {r['err']}, tools {r['tools']}, prompt {r['prompt']} (cache {r['cached']}), "
              f"text {r['text'].replace(chr(10), ' ')[:160]!r}", flush=True)

    for i in range(1, a.repeat + 1):
        r = ask(f"disk-{i}", "How much free disk space does the system have?")
        check(f"disk-{i}", r, not r["err"] and "disk_free" in r["tools"] and set(r["tools"]) <= {"disk_free"}
              and any("123.4 GB free" in x for x in r["results"]) and "123.4" in r["text"],
              "disk_free called, nothing else, its result stored, 123.4 in the answer")
        f = ask(f"followup-{i}", "According to the tool you used, how many gigabytes were free? Reply with the "
                "number only.", after=r)
        # stored: the chat holds both turns, two messages each
        check(f"followup-{i}", f, not f["err"] and not f["tools"] and "123.4" in f["text"] and f["stored"] == 4,
              "the same chat's second turn: no tool call, 123.4 from the history")
    for i in range(1, a.repeat + 1):
        r = ask(f"note-{i}", "What is the secret word in my note called secret?")
        check(f"note-{i}", r, not r["err"] and "read_note" in r["tools"]
              and any("HELIOTROPE" in x for x in r["results"]) and "HELIOTROPE" in r["text"].upper(),
              "read_note called, its result stored, HELIOTROPE in the answer")
    iso = ask("isolation", "Have I told you a secret word in this conversation? If so, say it; if not, say no. "
              "Do not use any tools.")
    check("isolation", iso, not iso["err"] and bool(iso["done"]) and not iso["tools"]
          and "HELIOTROPE" not in iso["all"].upper(), "a new chat does not know another chat's word, and calls no tool")

    # The prefix cache through the client: a chat whose first message is long (about 7,800 tokens here).
    # What another chat can have in common with it is the system prompt and the tool list, which is
    # less than the isolation chat's whole prompt. A second turn that reuses more than that reuses
    # the long message, and the inventory is this run's own (its batch number leads the message and
    # salts every line), so only this chat's first turn can have left it on the server.
    # Two wordings that sent the model to Open WebUI's built-in tools instead (2026-11-17): "keep it
    # for my next question" (all 300 lines written to a note with write_note, 13,700 tokens of
    # arguments) and "Depot inventory, batch N." as the opening (three searches of the knowledge base).
    salt = uuid.uuid4().int % 100000
    lines = [f"Shelf {n}: {(n * 37 + salt) % 911} crates of part {(n * 7919 + salt) % 10007:05d}, aisle "
             f"{(n * 13 + salt) % 47}." for n in range(1, 301)]
    lines[150] = "Shelf 151: the depot code is QUILLON-58."
    r = ask("cache", f"Here is the depot inventory (batch {salt}). What is the number of its last shelf? Reply with "
            "the number only.\n\n" + "\n".join(lines))
    check("cache", r, not r["err"] and bool(r["done"]) and "300" in r["text"], "a long first message read (300 shelves)")
    f = ask("cache-followup", "What is the depot code in the inventory? Reply with the code only.", after=r)
    shared = iso["prompt"]
    reused = isinstance(f["cached"], int) and isinstance(shared, int) and f["cached"] > shared
    check("cache-followup", f, not f["err"] and "QUILLON-58" in f["text"].upper() and reused,
          f"the code recalled, and {f['cached']} tokens of the prompt reused: more than the {shared} any other chat "
          "could share")

    # The background tasks (title, tags, follow-ups) are requests to the server too: let them end, so
    # the server's log is whole when this returns.
    t0, busy = time.time(), list(chats) if session else []
    while busy and time.time() - t0 < a.timeout:
        busy = [c for c in busy if (call("GET", f"/api/tasks/chat/{c}") or {}).get("task_ids")]
        if busy:
            time.sleep(0.5)
    if busy:
        ok = False
        print(f"FAIL tasks: {len(busy)} chats still had tasks running after {a.timeout} s")
    if session:
        titled = sum(1 for c in chats if (call("GET", f"/api/v1/chats/{c}") or {}).get("title") not in (None, "", "New Chat"))
        print(f"{len(chats)} chats; Open WebUI's title task named {titled}")
    else:
        print(f"{len(chats)} chats (no session: no background tasks)")

    (out / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    failed = [n for n, r in results.items() if not r["passed"]]
    print(f"{len(results) - len(failed)} of {len(results)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
