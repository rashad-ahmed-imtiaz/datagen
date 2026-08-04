from __future__ import annotations

import os

from scenario_data_factory.exceptions import DatabricksResourceError


def generation_job_id() -> str:
    job_id = os.getenv("SDF_GENERATION_JOB_ID")
    if not job_id:
        raise DatabricksResourceError(
            "MISSING_GENERATION_JOB",
            "The Databricks App is missing its generation job resource binding.",
            remediation="Bind the Lakeflow generation job as app resource generation-job.",
        )
    return job_id
