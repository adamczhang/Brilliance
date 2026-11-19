"""Brilliance: the chat-template check against a served model, standard library only.

  .toolchain\\python\\python.exe -I win\\tools\\template_check.py --url http://127.0.0.1:8000 \\
      --model qwen3.8-27b --hf D:\\models\\hf\\Qwen\\Qwen3.8-27B-FP8 --out out\\template\\corpus

1. The served container's chat template (/props) against Hugging Face's chat_template.jinja in --hf,
   byte for byte.
2. Seven conversations (CONVS: plain, a system prompt, several turns, tools, a tool result, thinking
   off, mixed scripts and emoji) rendered by the server (/apply-template) and tokenised as text and
   as messages (/tokenize), which must agree.
3. Each rendering written to --out as <name>.txt and its ids as little-endian int32 to --out\\hf\\
   (<name>.ids and <name>.txt.ids), the layout rad-tokcheck --hf-dir reads, so Hugging Face's
   tokenizer.json can be held to them; template_hf_compare.py holds Hugging Face's own rendering and
   tokeniser to them.
"""
import argparse
import json
import pathlib
import struct
import sys
import urllib.request

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
CONVS = {
    "plain": {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
    "system": {"messages": [{"role": "system", "content": "You answer in one word."},
                            {"role": "user", "content": "Name a primary colour."}]},
    "multiturn": {"messages": [{"role": "user", "content": "Hi! Who are you?"},
                               {"role": "assistant", "content": "I am a helpful assistant."},
                               {"role": "user", "content": "Write a haiku about rain.\nThree lines."}]},
    "tools": {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER]},
    "toolresult": {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps({"city": "Paris", "unit": "c"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 14, \"sky\": \"light rain\"}"}],
        "tools": [WEATHER]},
    "nothink": {"messages": [{"role": "user", "content": "2+2?"}],
                "chat_template_kwargs": {"enable_thinking": False}},
    "unicode": {"messages": [{"role": "user", "content": "Traduis : « le modèle 模型 est prêt » 🚀 — merci"}]},
}


def call(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf", required=True, help="the Hugging Face model folder with chat_template.jinja")
    ap.add_argument("--out", required=True, help="where the renderings and their ids go")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = pathlib.Path(a.out)
    tpl = call(a.url, "/props")["chat_template"].encode("utf-8")
    hf = (pathlib.Path(a.hf) / "chat_template.jinja").read_bytes()
    print(f"template: container {len(tpl)} bytes, Hugging Face {len(hf)} bytes, identical {tpl == hf}")
    ok = tpl == hf
    (out / "hf").mkdir(parents=True, exist_ok=True)
    for name, conv in CONVS.items():
        body = dict(conv, model=a.model)
        text = call(a.url, "/apply-template", body)["prompt"]
        (out / f"{name}.txt").write_bytes(text.encode("utf-8"))
        as_text = call(a.url, "/tokenize", {"model": a.model, "content": text})["tokens"]
        as_msgs = call(a.url, "/tokenize", dict(body, add_generation_prompt=True))["tokens"]
        same = as_text == as_msgs
        ok &= same
        blob = struct.pack(f"<{len(as_text)}i", *as_text)
        for n in (f"{name}.txt.ids", f"{name}.ids"):
            (out / "hf" / n).write_bytes(blob)
        print(f"{name}: {len(text)} chars, {len(as_text)} tokens; text and messages tokenise alike {same}")
    print("every check passed" if ok else "a check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
