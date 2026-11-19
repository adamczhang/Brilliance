"""Brilliance (the sampling investigation): A logging pass-through: POST/GET on --listen go to --upstream; each POST body is saved as
<out>/<n>.json before it is forwarded, and the response streams back unchanged."""
import argparse
import http.server
import itertools
import pathlib
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=8101)
ap.add_argument("--upstream", default="http://127.0.0.1:8100")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
seq = itertools.count(1)


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def fwd(self, body):
        req = urllib.request.Request(a.upstream + self.path, data=body, method=self.command)
        for k in ("Content-Type", "Accept", "Authorization"):
            if self.headers.get(k):
                req.add_header(k, self.headers[k])
        try:
            r = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            r = e
        self.send_response(r.status if hasattr(r, "status") else r.code)
        for k, v in r.headers.items():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.close_connection = True

    def do_GET(self):
        self.fwd(None)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        (out / f"{next(seq):04d}.json").write_bytes(body)
        self.fwd(body)

    def log_message(self, *args):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", a.listen), H).serve_forever()
