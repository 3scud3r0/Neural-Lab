"""Painel local da v2 e console interativo da VM limitada por combustível."""
from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from .common import json_default
from .vm import Limits, Machine, assemble, compile_source


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def compact_report(report):
    if report is None: return None
    return {"version": report["version"], "checks": report["checks"], "device": report["device"],
            "elapsed_seconds": report["elapsed_seconds"], "all_checks_passed": report["all_checks_passed"],
            "programming": {key: report["programming"][key] for key in ("solved", "total", "tasks", "heldout_scope")},
            "synthesis": {"solved_test": report["synthesis"]["solved_test"],
                          "final": report["synthesis"]["history"][-1], "test": report["synthesis"]["test"]},
            "communication": report["communication"],
            "morphology": report["morphology"],
            "world": {"evaluation": report["world"]["evaluation"], "morphology_events": report["world"]["morphology_events"]},
            "neat": {key: report["neat"][key] for key in ("nodes", "connections", "validation_mse", "structural_variation_observed")},
            "self_modification": report["self_modification"]}


def handler_for(directory):
    directory = Path(directory).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def respond(self, value, status=200, html=False):
            payload = value if html else json.dumps(value, ensure_ascii=False, allow_nan=False, default=json_default).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                self.respond(Path(__file__).with_name("dashboard.html").read_bytes(), html=True)
            elif route == "/api/state":
                self.respond({"stage": read_json(directory / "latest.json"),
                              "world": read_json(directory / "world/latest.json"),
                              "report": compact_report(read_json(directory / "report.json")),
                              "environment": read_json(directory / "environment.json")})
            elif route == "/api/programs":
                programs = []
                for path in sorted((directory / "programming/generated").glob("*.ovm"))[:64]:
                    if path.stat().st_size <= 65536:
                        programs.append({"name": path.stem, "source": path.read_text(encoding="utf-8")})
                discovered = directory / "synthesis/discovered.ovm"
                if discovered.is_file() and discovered.stat().st_size <= 65536:
                    programs.append({"name": "genetic_discovery", "source": discovered.read_text(encoding="utf-8")})
                self.respond(programs)
            else:
                self.respond({"error": "not_found"}, 404)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/vm":
                self.respond({"error": "not_found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 65536:
                    self.respond({"error": "Use de 1 a 65536 bytes"}, 413)
                    return
                request = json.loads(self.rfile.read(length))
                fuel = request.get("fuel", 4000)
                if type(fuel) is not int or not 1 <= fuel <= 4000: raise ValueError("Fuel deve ser inteiro entre 1 e 4000")
                source = request["source"]
                if not isinstance(source, str): raise ValueError("Fonte precisa ser texto")
                inputs, array = request.get("inputs", []), request.get("array", [])
                if not isinstance(inputs, list) or not isinstance(array, list): raise ValueError("Entradas precisam ser listas")
                program = compile_source(source) if request.get("compile", False) else assemble(source)
                result = Machine(Limits(fuel=fuel)).run(program, inputs, array, trace=bool(request.get("trace", False)))
                self.respond(asdict(result))
            except (ValueError, TypeError, KeyError, SyntaxError, UnicodeDecodeError, RecursionError) as exc:
                self.respond({"ok": False, "fault": "request", "message": str(exc)[:300]}, 400)

    return Handler


def serve(directory, host="127.0.0.1", port=8080):
    server = ThreadingHTTPServer((host, port), handler_for(directory))
    print(f"Ouroboros 2: painel na porta {server.server_port}; dados em {Path(directory).resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
