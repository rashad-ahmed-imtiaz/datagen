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


def test_agent_draft_rejects_blank_and_oversized_prompts_without_calling_model() -> None:
    client = TestClient(app)
    assert client.post("/api/agent/draft", json={"prompt": "   "}).status_code == 422
    assert client.post("/api/agent/draft", json={"prompt": "x" * 20_001}).status_code == 422


def test_missing_scenario_is_a_clean_not_found_response() -> None:
    response = TestClient(app).get("/api/scenarios/scn_missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "scenario not found"


def test_missing_run_is_a_clean_not_found_response() -> None:
    response = TestClient(app).get("/api/runs/run_missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "scenario or run not found"
