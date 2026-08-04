from __future__ import annotations

import sys
from pathlib import Path

from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.agent_server.agent  # noqa: F401
from app.main import app as api_app

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app
app.include_router(api_app.router)
setup_mlflow_git_based_version_tracking()


def main() -> None:
    agent_server.run(app_import_string="app.start_server:app")


if __name__ == "__main__":
    main()
