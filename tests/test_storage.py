import importlib
import os
import tempfile


def test_storage_save_and_fetch(monkeypatch):
    # Use a temp DB path before importing db module
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("DATABASE_PATH", path)

    # Re-import config and db to pick up new env
    from app import config as cfg

    importlib.reload(cfg)
    from app.storage import db as sdb

    importlib.reload(sdb)

    sid = "testsession"
    prompt = "Hello"
    code = "print('hi')"
    diag = "YmFzZTY0"  # fake
    analysis = {"token_count": 2}

    mid = sdb.save_message(sid, prompt, code, diag, analysis)
    assert isinstance(mid, int)

    hist = sdb.fetch_history(sid)
    assert len(hist) == 1
    item = hist[0]
    assert item["user_prompt"] == prompt
    assert item["generated_code"] == code


