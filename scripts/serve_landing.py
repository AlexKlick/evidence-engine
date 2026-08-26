#!/usr/bin/env python3
"""Read-only allowlist static server for a landing experiment (ADR-0005).

Serves EXACTLY the A/B landing pages, the events contract, and /healthz —
nothing else exists to traverse, list, or write. GET/HEAD only (405
otherwise). Binds loopback only; the public hub proxies it under a
configured route (the concrete profile name lives in the operator's hub
route source, not this repo), mirroring the fractal-page open-static-public
pattern. No secrets, no writes, no capability of any kind beyond serving
three static files.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ALLOWED_FILES = {
    "landing-a.html": "text/html; charset=utf-8",
    "landing-b.html": "text/html; charset=utf-8",
    "events.json": "application/json; charset=utf-8",
}


class LandingHandler(BaseHTTPRequestHandler):
    server_version = "evidence-landing/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _serve(self) -> None:
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        root = Path(self.server.directory)  # type: ignore[attr-defined]

        if path in ("", "/"):
            variants = "".join(
                f'<li><a href="{name}">{name}</a></li>' for name in ALLOWED_FILES
                if name.endswith(".html")
            )
            body = (
                "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<title>Landing experiment</title></head>"
                "<body><h1>Landing experiment</h1>"
                "<p>Price-test variants (visible price before the CTA):</p>"
                f"<ul>{variants}</ul></body></html>"
            ).encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            return

        if path == "/healthz":
            body = json.dumps({"ok": True, "service": "evidence-landing"}).encode()
            self._send(200, body, "application/json; charset=utf-8")
            return

        name = path.lstrip("/")
        if name in ALLOWED_FILES:
            target = root / name
            try:
                body = target.read_bytes()
            except OSError:
                self._send(404, b"not found\n", "text/plain; charset=utf-8")
                return
            self._send(200, body, ALLOWED_FILES[name])
            return

        # anything not on the allowlist — including traversal attempts — 404s
        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        self._serve()

    def do_HEAD(self) -> None:  # noqa: N802 (http.server naming)
        self._serve()

    def _reject(self) -> None:
        self._send(405, b"method not allowed\n", "text/plain; charset=utf-8")

    do_POST = do_PUT = do_DELETE = do_PATCH = _reject  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory", required=True, help="directory holding the landing files"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18087)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), LandingHandler)
    server.directory = args.directory  # type: ignore[attr-defined]
    print(
        f"evidence-landing serving {args.directory} on {args.host}:{args.port} "
        f"(allowlist: {', '.join(sorted(ALLOWED_FILES))}, /healthz)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
