import os
import sys

# Ensure the backend does not perform heavy initialization during tests
os.environ["OMNIMESH_TESTING"] = "1"
os.environ["SKIP_MODEL_DOWNLOAD"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omnimesh.backend import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_login():
    resp = client.post("/token", json={"password": "secure_omnimesh_pass_2025"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()
