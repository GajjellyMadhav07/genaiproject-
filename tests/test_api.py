import importlib
import os
import tempfile
import base64

import pytest


@pytest.fixture()
def app_client(monkeypatch):
    # Use a temp DB path before importing server module
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("DATABASE_PATH", path)

    # Mock async services to avoid external calls
    async def fake_generate_code(prompt: str) -> str:
        return "print('ok')\n"

    async def fake_generate_diagram(prompt: str, code: str | None) -> bytes:
        return base64.b64decode(base64.b64encode(b"img"))

    from app.services import codegen as cg
    from app.services import diagram as dg

    monkeypatch.setattr(cg, "generate_code", fake_generate_code)
    monkeypatch.setattr(dg, "generate_diagram", fake_generate_diagram)

    # Reload config (pick env) and server
    from app import config as cfg

    importlib.reload(cfg)
    from app.api import server as api_server

    importlib.reload(api_server)
    app = api_server.create_app()
    with app.test_client() as client:
        yield client


def test_health(app_client):
    resp = app_client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_generate_and_history(app_client):
    payload = {"prompt": "Say hi", "session_id": "s1"}
    resp = app_client.post("/api/generate", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "generated_code" in data
    assert "diagram_base64" in data
    assert "analysis" in data

    # Then fetch history
    hist = app_client.get("/api/history?session_id=s1")
    assert hist.status_code == 200
    items = hist.get_json()
    assert isinstance(items, list)
    assert len(items) == 1
    assert items[0]["user_prompt"] == "Say hi"


