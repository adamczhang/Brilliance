"""Brilliance: a stand-in Hugging Face hub on loopback, for win\\tests\\fetch_launch_test.ps1.

Serves the repos under --root (ROOT/<owner>/<repo>/<files>) the way hfget.ps1 reads the hub:
  GET /api/models/<owner>/<repo>/revision/<rev>   {"sha": <--commit>}
  GET /api/models/<owner>/<repo>/tree/<rev>?...   the files, each with its size and LFS sha256
  GET /<owner>/<repo>/resolve/<rev>/<path>        the bytes, honouring "Range: bytes=N-"
Only the revision 'main' and the --commit sha are known; anything else is a 404.

--drop N cuts the first N file responses short: the headers promise the whole body, half of what
was asked is sent, and the connection is closed, as a dropped link does. Every request is logged
to --log, one line each ("<method> <path> <range>"), so a test can count transfers. The port it
listens on is written to --port-file once it listens. Standard library only.
"""
import argparse
import hashlib
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--port-file", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--commit", default="0123456789abcdef0123456789abcdef01234567")
ap.add_argument("--drop", type=int, default=0)
args = ap.parse_args()

lock = threading.Lock()
drops = [args.drop]


def log(line):
    with lock:
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def repo_dir(owner, repo):
    d = os.path.join(args.root, owner, repo)
    return d if os.path.isdir(d) else None


def known(rev):
    return rev in ("main", args.commit)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Hub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def not_found(self):
        self.send_json({"error": "not found"}, 404)

    def do_GET(self):
        u = urlsplit(self.path)
        parts = [unquote(p) for p in u.path.split("/") if p]
        log("GET %s %s" % (u.path, self.headers.get("Range", "-")))
        if len(parts) >= 6 and parts[:2] == ["api", "models"] and parts[4] == "revision":
            rev = "/".join(parts[5:])
            if not repo_dir(parts[2], parts[3]) or not known(rev):
                return self.not_found()
            return self.send_json({"id": parts[2] + "/" + parts[3], "sha": args.commit})
        if len(parts) >= 6 and parts[:2] == ["api", "models"] and parts[4] == "tree":
            d = repo_dir(parts[2], parts[3])
            if not d or not known("/".join(parts[5:])):
                return self.not_found()
            out = []
            for base, _, names in os.walk(d):
                for n in sorted(names):
                    p = os.path.join(base, n)
                    rel = os.path.relpath(p, d).replace(os.sep, "/")
                    out.append({"type": "file", "path": rel, "size": os.path.getsize(p),
                                "lfs": {"oid": sha256(p), "size": os.path.getsize(p)}})
            return self.send_json(out)
        if len(parts) >= 5 and parts[2] == "resolve":
            d = repo_dir(parts[0], parts[1])
            if not d or not known(parts[3]):
                return self.not_found()
            p = os.path.join(d, *parts[4:])
            if not os.path.isfile(p):
                return self.not_found()
            data = open(p, "rb").read()
            start = 0
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes=") and rng[6:].endswith("-"):
                start = int(rng[6:-1])
            if start > len(data):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = data[start:]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            cut = False
            with lock:
                if drops[0] > 0 and len(body) > 1:
                    drops[0] -= 1
                    cut = True
            if cut:
                self.wfile.write(body[: len(body) // 2])
                self.wfile.flush()
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                log("DROP %s after %d of %d bytes" % (u.path, len(body) // 2, len(body)))
                return
            self.wfile.write(body)
            return
        self.not_found()


srv = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
with open(args.port_file, "w", encoding="utf-8") as f:
    f.write(str(srv.server_address[1]))
srv.serve_forever()
