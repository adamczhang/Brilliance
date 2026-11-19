"""Brilliance: the chat-template check against Hugging Face's own code.

  python -I win\\tools\\template_hf_compare.py --hf D:\\models\\hf\\Qwen\\Qwen3.8-27B-FP8 \\
      --corpus out\\template\\corpus

template_check.py's seven conversations, rendered by transformers' apply_chat_template over the
model's own tokenizer files and tokenised by the tokenizers library, set against what the server
rendered (<corpus>\\<name>.txt) and its ids (<corpus>\\hf\\<name>.ids): text and ids must be identical.
Needs transformers, tokenizers and jinja2 (not PyTorch) in the Python that runs it, e.g. a venv; it
reads only local files (HF_HUB_OFFLINE). Run template_check.py against the server first.
"""
import argparse
import json
import os
import pathlib
import struct
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from template_check import CONVS  # noqa: E402  (the same conversations, same order)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", required=True, help="the Hugging Face model folder (tokenizer files, template)")
    ap.add_argument("--corpus", required=True, help="template_check.py's --out")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf, local_files_only=True)
    corpus = pathlib.Path(a.corpus)
    ok = True
    for name, conv in CONVS.items():
        server_text = (corpus / f"{name}.txt").read_bytes().decode("utf-8")
        raw = (corpus / "hf" / f"{name}.ids").read_bytes()
        server_ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
        kw = dict(conv.get("chat_template_kwargs", {}))
        # The OpenAI wire carries a tool call's arguments as a JSON string; Hugging Face's templates
        # take the parsed mapping (the server parses it before rendering, as vLLM does).
        msgs = json.loads(json.dumps(conv["messages"]))
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
        hf_text = tok.apply_chat_template(msgs, tools=conv.get("tools"), add_generation_prompt=True,
                                          tokenize=False, **kw)
        hf_ids = tok.encode(hf_text, add_special_tokens=False)
        same_text, same_ids = hf_text == server_text, hf_ids == server_ids
        ok &= same_text and same_ids
        detail = ""
        if not same_text:
            i = next((k for k in range(min(len(hf_text), len(server_text))) if hf_text[k] != server_text[k]),
                     min(len(hf_text), len(server_text)))
            detail = f"; first text difference at char {i}: HF {hf_text[i:i + 40]!r} / server {server_text[i:i + 40]!r}"
        elif not same_ids:
            i = next((k for k in range(min(len(hf_ids), len(server_ids))) if hf_ids[k] != server_ids[k]),
                     min(len(hf_ids), len(server_ids)))
            detail = f"; first id difference at {i}: HF {hf_ids[i:i + 6]} / server {server_ids[i:i + 6]}"
        print(f"{name}: text {'identical' if same_text else 'DIFFERS'} ({len(hf_text)} chars), "
              f"ids {'identical' if same_ids else 'DIFFER'} ({len(hf_ids)} HF, {len(server_ids)} server){detail}")
    print("every conversation identical to Hugging Face's rendering and tokenisation" if ok
          else "a difference FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
