from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


PAGE = Path(__file__).with_name("dashboard.html")


def handler_for(output):
    output = Path(output).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = urlsplit(self.path).path
            status = 200
            if path == "/":
                content, mime = PAGE.read_bytes(), "text/html; charset=utf-8"
            elif path in ("/api/latest", "/api/history"):
                try:
                    if path == "/api/latest":
                        value = json.loads((output / "latest.json").read_text(encoding="utf-8"))
                    else:
                        with (output / "metrics.jsonl").open(encoding="utf-8") as stream:
                            lines = deque(stream, maxlen=300)
                        value = []
                        for line in lines:
                            try:
                                value.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass  # Última linha pode estar sendo gravada.
                    content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                except (FileNotFoundError, json.JSONDecodeError):
                    status, content = 503, b'{"message":"Aguardando a simulacao"}'
                mime = "application/json; charset=utf-8"
            else:
                status, content, mime = 404, b"Not found", "text/plain"
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def serve(output, host="127.0.0.1", port=8080):
    server = ThreadingHTTPServer((host, port), handler_for(output))
    print(f"Painel ouvindo na porta {server.server_port}; dados: {Path(output).resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
