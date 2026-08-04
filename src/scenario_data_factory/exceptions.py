from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScenarioDataFactoryError(Exception):
    error_code: str
    user_message: str
    technical_detail: str = ""
    scenario_id: str | None = None
    run_id: str | None = None
    remediation: str | None = None

    def __str__(self) -> str:
        detail = f" ({self.technical_detail})" if self.technical_detail else ""
        return f"{self.error_code}: {self.user_message}{detail}"


class ScenarioValidationError(ScenarioDataFactoryError):
    pass


class ScenarioRevisionConflict(ScenarioDataFactoryError):
    pass


class UnsupportedBlueprintError(ScenarioDataFactoryError):
    pass


class UnsupportedIssueError(ScenarioDataFactoryError):
    pass


class DatagenCompilationError(ScenarioDataFactoryError):
    pass


class BaselineValidationError(ScenarioDataFactoryError):
    pass


class IssuePlanningError(ScenarioDataFactoryError):
    pass


class IssueInjectionError(ScenarioDataFactoryError):
    pass


class OutputWriteError(ScenarioDataFactoryError):
    pass


class RunStateError(ScenarioDataFactoryError):
    pass


class DatabricksResourceError(ScenarioDataFactoryError):
    pass


class DependencyResolutionError(ScenarioDataFactoryError):
    pass
