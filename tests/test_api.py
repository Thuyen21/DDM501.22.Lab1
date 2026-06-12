# tests/test_api.py - automated tests for the API with pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_valid():
    payload = {
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ["setosa", "versicolor", "virginica"]
    assert 0.0 <= body["confidence"] <= 1.0

def test_predict_invalid_type():
    # send text instead of a number -> must be rejected with 422
    r = client.post("/predict", json={"sepal_length": "abc"})
    assert r.status_code == 422
