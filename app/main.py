from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from scenario_data_factory.app_services.scenario_service import AgentPlanningError, ScenarioService
from scenario_data_factory.exceptions import ScenarioDataFactoryError, ScenarioRevisionConflict

app = FastAPI(title="Scenario Data Factory")
service = ScenarioService()
_DRAFT_TIMEOUT_SECONDS = 120


class DraftRequest(BaseModel):
    domain: str = "insurance_claims"
    name: str
    seed: int = 42
    scale: str = "demo"


class PromptDraftRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)

    @field_validator("prompt")
    @classmethod
    def non_blank_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt cannot be blank")
        return value


class PatchRequest(BaseModel):
    expected_revision: int
    patch: dict[str, object]


class ConfirmationRequest(BaseModel):
    confirmation_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


@app.exception_handler(ScenarioRevisionConflict)
async def handle_revision_conflict(_, exc: ScenarioRevisionConflict) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ScenarioDataFactoryError)
async def handle_sdf_error(_, exc: ScenarioDataFactoryError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def handle_missing_resource(_, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "scenario or run not found"})


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/agent/draft")
async def create_scenario_from_prompt(request: PromptDraftRequest) -> dict[str, object]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(service.create_scenario_from_prompt, request.prompt),
            timeout=_DRAFT_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "The scenario planner did not finish within two minutes. "
                "No data or tables were created; please submit the request again."
            ),
        ) from exc
    except AgentPlanningError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        print("Unexpected scenario draft validation failure:", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "The scenario planner could not complete an executable draft. "
                "No data or tables were created; please submit the request again."
            ),
        ) from exc


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


@app.get("/api/runs/{run_id}")
async def get_run_summary(run_id: str) -> dict[str, object]:
    return service.get_run_summary(run_id)


@app.post("/responses-compatible")
async def responses_compatible(payload: dict[str, object]) -> dict[str, object]:
    from app.agent_server.agent import invoke_agent_dict

    return await invoke_agent_dict(payload)
