from __future__ import annotations

from scenario_data_factory.app_services.scenario_service import (
    _dataset_code_from_intent,
    _raw_runs_root,
    _workspace_output_location,
)
from scenario_data_factory.models.scenario import OutputSpec
from scenario_data_factory.output.delta_writer import write_delta_tables


class _Writer:
    def __init__(self, saved: list[str]) -> None:
        self._saved = saved

    def format(self, _format: str) -> _Writer:
        return self

    def mode(self, _mode: str) -> _Writer:
        return self

    def option(self, _key: str, _value: str) -> _Writer:
        return self

    def saveAsTable(self, name: str) -> None:
        self._saved.append(name)


class _DataFrame:
    def __init__(self, saved: list[str]) -> None:
        self.write = _Writer(saved)


def test_default_delta_names_use_entity_then_quality_suffix() -> None:
    outputs = OutputSpec()
    assert outputs.clean_delta_prefix == ""
    assert outputs.dirty_delta_prefix == "dq"

    clean_written = write_delta_tables(
        {"customers": _DataFrame([])}, "sdf", "scenario_data_factory", outputs.clean_delta_prefix,
        namespace="cbank",
    )
    dirty_saved: list[str] = []
    dirty_written = write_delta_tables(
        {"customers": _DataFrame(dirty_saved)},
        "sdf",
        "scenario_data_factory",
        outputs.dirty_delta_prefix,
        namespace="cbank",
    )

    assert clean_written == ["`sdf`.`scenario_data_factory`.`cbank_customers`"]
    assert dirty_written == ["`sdf`.`scenario_data_factory`.`cbank_customers_dq`"]
    assert dirty_saved == ["`sdf`.`scenario_data_factory`.`cbank_customers_dq`"]


def test_agent_dataset_code_controls_delta_namespace() -> None:
    assert _dataset_code_from_intent({"dataset_code": "cbank"}, "Canadian Banking") == "cbank"
    assert _dataset_code_from_intent({}, "Canadian Banking") == "canadian_banking"


def test_bound_volumes_define_the_agent_output_location(monkeypatch) -> None:
    monkeypatch.setenv(
        "SDF_CONTROL_VOLUME",
        "/Volumes/team_sandbox/dev_alex_scenario_data_factory/sdf_control",
    )
    monkeypatch.setenv(
        "SDF_RAW_VOLUME",
        "/Volumes/team_sandbox/dev_alex_scenario_data_factory/sdf_raw",
    )

    assert _workspace_output_location() == ("team_sandbox", "dev_alex_scenario_data_factory")
    assert _raw_runs_root() == "/Volumes/team_sandbox/dev_alex_scenario_data_factory/sdf_raw/runs"
