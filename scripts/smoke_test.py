import os
import sys
import json
import urllib.request
import urllib.error


def http_json(method: str, url: str, body: dict | None = None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if e.fp else ""
        try:
            return e.code, json.loads(raw) if raw else None
        except Exception:
            return e.code, raw


def main():
    # Ensure project root is importable when running from scripts/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("== Import checks ==")
    import backend.main as main_mod

    assert main_mod.app is not None
    print("backend.main: OK")

    from backend.agent.ai_agent import agent

    assert agent is not None
    print("backend.agent.ai_agent: OK")

    print("\n== Environment ==")
    print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))

    print("\n== API checks (expects backend running on 127.0.0.1:8000) ==")
    base = "http://127.0.0.1:8000"

    status, data = http_json("GET", f"{base}/")
    print("GET /", status, data)

    status, data = http_json("POST", f"{base}/chat", {"message": "hello"})
    print("POST /chat", status, data)

    print("\nSmoke test complete.")


if __name__ == "__main__":
    raise SystemExit(main())

