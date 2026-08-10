from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from scenario_data_factory.app_services.scenario_service import ScenarioService

app = FastAPI(title="Scenario Data Factory")
service = ScenarioService()


class DraftRequest(BaseModel):
    domain: str = "insurance_claims"
    name: str
    seed: int = 42
    scale: str = "demo"


class PromptDraftRequest(BaseModel):
    prompt: str


class PatchRequest(BaseModel):
    expected_revision: int
    patch: dict[str, object]


class ConfirmationRequest(BaseModel):
    confirmation_hash: str


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/agent/draft")
async def create_scenario_from_prompt(request: PromptDraftRequest) -> dict[str, object]:
    try:
        return service.create_scenario_from_prompt(request.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/scenarios")
async def create_scenario(request: DraftRequest) -> dict[str, object]:
    return service.create_scenario_draft(request.model_dump())


@app.get("/api/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str) -> dict[str, object]:
    try:
        return service.get_scenario_draft(scenario_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="scenario not found") from exc


@app.patch("/api/scenarios/{scenario_id}")
async def patch_scenario(scenario_id: str, request: PatchRequest) -> dict[str, object]:
    return service.patch_scenario_draft(scenario_id, request.expected_revision, request.patch)


@app.post("/api/scenarios/{scenario_id}/validate")
async def validate_scenario(scenario_id: str) -> dict[str, object]:
    return service.validate_scenario_draft(scenario_id)


@app.post("/api/scenarios/{scenario_id}/preview")
async def prepare_preview(scenario_id: str) -> dict[str, object]:
    return service.prepare_preview(scenario_id)


@app.post("/api/scenarios/{scenario_id}/generation")
async def prepare_generation(scenario_id: str) -> dict[str, object]:
    return service.prepare_generation(scenario_id)


@app.post("/api/generation/{run_id}/confirm")
async def confirm_generation(run_id: str, request: ConfirmationRequest) -> dict[str, object]:
    return service.confirm_generation(run_id, request.confirmation_hash)


@app.post("/api/generation/{run_id}/submit")
async def submit_generation(run_id: str, request: ConfirmationRequest) -> dict[str, object]:
    return service.confirm_and_submit_generation(run_id, request.confirmation_hash)


@app.post("/responses-compatible")
async def responses_compatible(payload: dict[str, object]) -> dict[str, object]:
    from app.agent_server.agent import invoke_agent_dict

    return await invoke_agent_dict(payload)
