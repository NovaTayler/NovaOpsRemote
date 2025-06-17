import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from novadash.main import app, config


def test_index():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "NovaDash" in resp.get_data(as_text=True)


def test_send_command(tmp_path):
    client = app.test_client()
    config.log_path = tmp_path / "log.txt"
    resp = client.post("/send", data={"command": "echo hello"})
    assert resp.status_code == 200
    assert "hello" in resp.get_data(as_text=True)
    log_text = config.log_path.read_text()
    assert "CMD: echo hello" in log_text
