from fastapi.testclient import TestClient

from app.main import app


def test_root_page_renders() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Scenario Data Factory" in response.text


def test_health_route() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
