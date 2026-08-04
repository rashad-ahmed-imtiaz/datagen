from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from app.agent_server.instructions import AGENT_INSTRUCTIONS
from app.agent_server.tools import tool_registry

try:  # pragma: no cover - depends on Databricks App MLflow runtime
    from mlflow.genai.agent_server import invoke, stream
except Exception:  # pragma: no cover

    def invoke():
        def wrapper(fn):
            return fn

        return wrapper

    def stream():
        def wrapper(fn):
            return fn

        return wrapper


async def _run_openai_agent(payload: dict[str, Any]) -> str:
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        raise RuntimeError("SDF_MODEL_ENDPOINT is missing; bind a model serving endpoint resource.")
    try:  # pragma: no cover - exercised in Databricks App runtime
        from agents import Agent, Runner
    except Exception:
        return (
            "Scenario Data Factory agent runtime is not installed locally. "
            "The deterministic API and CLI remain available."
        )

    agent = Agent(
        name="Scenario Data Factory",
        instructions=AGENT_INSTRUCTIONS,
        model=endpoint,
        tools=tool_registry(),
    )
    user_input = payload.get("input") or payload.get("messages") or ""
    result = await Runner.run(agent, str(user_input))
    return str(result.final_output)


@invoke()
async def invoke_agent(request: Any) -> Any:
    payload = request.model_dump() if hasattr(request, "model_dump") else dict(request)
    content = await _run_openai_agent(payload)
    response = {"output": [{"type": "message", "role": "assistant", "content": content}]}
    try:
        from mlflow.types.responses import ResponsesAgentResponse
    except Exception:
        return response
    return ResponsesAgentResponse(**response)


@stream()
async def stream_agent(request: Any) -> AsyncIterator[dict[str, Any]]:
    payload = request.model_dump() if hasattr(request, "model_dump") else dict(request)
    content = await _run_openai_agent(payload)
    yield {"type": "response.output_text.delta", "delta": content}
    yield {"type": "response.completed"}


async def invoke_agent_dict(payload: dict[str, Any]) -> dict[str, Any]:
    content = await _run_openai_agent(payload)
    return {"output": [{"type": "message", "role": "assistant", "content": content}]}
