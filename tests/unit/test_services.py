from __future__ import annotations

import scenario_data_factory.app_services.scenario_service as scenario_service
from scenario_data_factory.app_services.scenario_service import ScenarioService, _json_from_text
from scenario_data_factory.persistence.run_repository import RunRepository
from scenario_data_factory.persistence.scenario_repository import ScenarioRepository


def test_generation_confirmation_hash_guard(tmp_path) -> None:
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )
    draft = service.create_scenario_draft(
        {"domain": "insurance_claims", "name": "demo", "seed": 42, "scale": "small"}
    )
    prepared = service.prepare_generation(str(draft["scenario_id"]))
    rejected = service.confirm_generation(str(prepared["run_id"]), "wrong")
    assert rejected["status"] == "rejected"
    confirmed = service.confirm_generation(str(prepared["run_id"]), str(draft["spec_hash"]))
    assert confirmed["status"] == "confirmed"


def test_create_scenario_from_prompt_parses_counts_and_issues(tmp_path) -> None:
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build retail data with 2k customers, 5k orders, duplicate orders, "
        "3% missing email, and orphan customer ids."
    )

    assert result["domain"] == "retail_orders"
    assert result["tables"]["customers"] == 2_000
    assert result["tables"]["orders"] == 5_000
    assert any(issue["type"] == "duplicate_record" for issue in result["issues"])
    assert any(issue["column"] == "email" for issue in result["issues"])


def test_agent_table_counts_drive_default_insurance_issue_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_agent_intent_from_model",
        lambda prompt: (
            {
                "domain": "insurance_claims",
                "name": "Agent Insurance Plan",
                "scale": "demo",
                "seed": 42,
                "table_counts": {
                    "customers": 16_667,
                    "policies": 25_000,
                    "claims": 50_000,
                    "payments": 66_667,
                },
                "issues": [],
            },
            "Model extracted scenario intent.",
        ),
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt("Build insurance data with 50k claims.")
    issue_counts = {issue["issue_id"]: issue["exact_count"] for issue in result["issues"]}

    assert result["tables"]["claims"] == 50_000
    assert result["tables"]["customers"] == 16_667
    assert result["tables"]["policies"] == 25_000
    assert result["tables"]["payments"] == 66_667
    assert issue_counts["iss_duplicate_claims"] == 1_500
    assert issue_counts["iss_policy_orphans"] == 500
    assert issue_counts["iss_late_payments"] == 2_667


def test_agent_plan_and_explicit_issue_rates_win(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_agent_intent_from_model",
        lambda prompt: (
            {
                "domain": "insurance_claims",
                "name": "Agent Insurance Plan",
                "scale": "demo",
                "seed": 42,
                "table_counts": {
                    "customers": 33_333,
                    "policies": 50_000,
                    "claims": 100_000,
                    "payments": 133_333,
                },
                "issues": [],
            },
            "Model extracted scenario intent.",
        ),
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build insurance claims with 100k claims, duplicate claims, "
        "2% orphan policy ids, 5% missing adjusters, 10% late payments, "
        "schema drift, replayed files, and invalid customer provinces."
    )
    preview = service.prepare_preview(str(result["scenario_id"]))

    assert result["tables"] == {
        "customers": 33_333,
        "policies": 50_000,
        "claims": 100_000,
        "payments": 133_333,
    }
    assert preview["summary"]["issue_counts"]["iss_policy_orphans"] == 2_000
    assert preview["summary"]["issue_counts"]["iss_late_payments"] == 13_333


def test_incomplete_agent_issue_does_not_trigger_full_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_agent_intent_from_model",
        lambda prompt: (
            {
                "domain": "insurance_claims",
                "name": "Agent Insurance Plan",
                "scale": "demo",
                "seed": 42,
                "table_counts": {
                    "customers": 50_000,
                    "policies": 120_000,
                    "claims": 100_000,
                    "payments": 110_000,
                },
                "issues": [
                    {
                        "type": "duplicate_record",
                        "table": "claims",
                        "column": None,
                        "parameters": {},
                    }
                ],
            },
            "Model extracted scenario intent.",
        ),
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt("Build insurance data with 100k claims.")

    assert result["tables"] == {
        "customers": 50_000,
        "policies": 120_000,
        "claims": 100_000,
        "payments": 110_000,
    }
    assert result["issues"][0]["rate"] == 0.01
    assert not any("deterministic fallback" in warning for warning in result["warnings"])


def test_truncated_model_json_returns_none() -> None:
    assert _json_from_text('{"table_counts": {"claims": 50000}, "issues": [') is None


def test_custom_schema_intent_is_completed_by_agent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_agent_intent_from_model",
        lambda prompt: (
            {"domain": "custom_schema", "name": "Healthcare Claims", "scale": "demo"},
            "Model extracted scenario intent.",
        ),
    )
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: {
            "domain": "custom_schema",
            "name": "Healthcare Claims",
            "scale": "demo",
            "seed": 42,
            "table_specs": [
                {
                    "name": "patients",
                    "row_count": 50_000,
                    "columns": [
                        {
                            "name": "patient_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "province", "type": "string"},
                    ],
                },
                {
                    "name": "providers",
                    "row_count": 5_000,
                    "columns": [
                        {
                            "name": "provider_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        }
                    ],
                },
                {
                    "name": "claims",
                    "row_count": 100_000,
                    "columns": [
                        {
                            "name": "claim_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "patient_id", "type": "long", "nullable": False},
                        {"name": "provider_id", "type": "long", "nullable": False},
                    ],
                },
            ],
            "relationships": [
                {
                    "name": "patients_claims",
                    "parent_table": "patients",
                    "parent_column": "patient_id",
                    "child_table": "claims",
                    "child_column": "patient_id",
                }
            ],
            "issues": [
                {
                    "type": "referential_orphan",
                    "table": "claims",
                    "column": "patient_id",
                    "rate": 0.02,
                    "exact_count": None,
                    "parameters": {},
                }
            ],
        },
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt("Build healthcare claims with bad data")

    assert result["domain"] == "custom_schema"
    assert result["tables"]["patients"] == 50_000
    assert result["tables"]["claims"] == 100_000
    assert result["issues"][0]["column"] == "patient_id"


def test_non_blueprint_domain_uses_agent_custom_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_agent_intent_from_model",
        lambda prompt: (
            {"domain": "healthcare_claims", "name": "Healthcare Claims", "scale": "demo"},
            "Model extracted scenario intent.",
        ),
    )
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: {
            "domain": "custom_schema",
            "name": "Healthcare Claims",
            "scale": "demo",
            "seed": 42,
            "table_specs": [
                {
                    "name": "patients",
                    "row_count": 50_000,
                    "columns": [
                        {
                            "name": "patient_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "province", "type": "string"},
                    ],
                },
                {
                    "name": "claims",
                    "row_count": 100_000,
                    "columns": [
                        {
                            "name": "claim_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "patient_id", "type": "long", "nullable": False},
                        {"name": "ingestion_ts", "type": "timestamp"},
                    ],
                },
            ],
            "relationships": [
                {
                    "name": "patients_claims",
                    "parent_table": "patients",
                    "parent_column": "patient_id",
                    "child_table": "claims",
                    "child_column": "patient_id",
                }
            ],
            "issues": [
                {
                    "type": "duplicate_record",
                    "table": "claims",
                    "column": None,
                    "rate": 0.01,
                    "exact_count": None,
                    "parameters": {},
                }
            ],
        },
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt("Build healthcare claims with 100k claims.")

    assert result["domain"] == "custom_schema"
    assert set(result["tables"]) == {"patients", "claims"}
    assert "customers" not in result["tables"]


def test_healthcare_heuristic_fallback_stays_custom_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_agent_intent_from_model",
        lambda prompt: (None, "Model unavailable."),
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build healthcare claims with 100k claims, duplicate claims, "
        "2% orphan patient IDs, 5% missing provider IDs, 10% late adjudications, "
        "schema drift, replayed files, and invalid patient provinces."
    )

    assert result["domain"] == "custom_schema"
    assert result["tables"]["claims"] == 100_000
    assert {"patients", "providers", "claims", "adjudications", "payments"}.issubset(
        set(result["tables"])
    )
    assert "customers" not in result["tables"]


def test_telecom_fallback_designs_domain_tables(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: None,
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build telecom network events with 100k events, duplicate events, "
        "2% orphan cell tower IDs, 5% missing network engineers, "
        "10% late incident closures, schema drift, replayed files, "
        "and invalid customer regions."
    )

    assert result["domain"] == "custom_schema"
    assert result["tables"]["network_events"] == 100_000
    assert {
        "customer_regions",
        "cell_towers",
        "network_engineers",
        "network_events",
        "incidents",
        "incident_closures",
    }.issubset(set(result["tables"]))
    issue_targets = {(issue["type"], issue["table"], issue["column"]) for issue in result["issues"]}
    assert ("referential_orphan", "network_events", "tower_id") in issue_targets
    assert ("null_value", "incidents", "assigned_engineer_id") in issue_targets
    assert ("late_arrival", "incident_closures", "closure_ts") in issue_targets
    assert ("invalid_value", "customer_regions", "region_code") in issue_targets
    issue_by_type = {issue["type"]: issue for issue in result["issues"]}
    assert issue_by_type["invalid_value"]["exact_count"] == 1
    assert issue_by_type["invalid_value"]["parameters"]["invalid_values"] == [
        "UNKNOWN",
        "12345",
        "",
        "REGION_@@",
    ]
    assert issue_by_type["file_replay"]["exact_count"] is None
    assert issue_by_type["file_replay"]["rate"] is None
    assert issue_by_type["file_replay"]["parameters"]["file_count"] == 1
    assert issue_by_type["file_replay"]["parameters"]["source_batch_label"] == "batch_002"
    assert issue_by_type["file_replay"]["parameters"]["replay_batch"] == "batch_004"
    assert issue_by_type["late_arrival"]["parameters"]["semantics"] == "late_arriving_data"
    assert issue_by_type["late_arrival"]["parameters"]["arrival_column"] == "ingestion_ts"
    assert issue_by_type["schema_drift"]["parameters"]["operation"] == "add_column"
    assert issue_by_type["schema_drift"]["parameters"]["column"] == {
        "name": "signal_strength",
        "type": "double",
    }
    assert result["timeline"]["batches"] >= 4
    assert issue_by_type["file_replay"]["display_value"] == "1 file"
    assert any(
        "one invalid region_code record rather than a percentage" in warning
        for warning in result["warnings"]
    )
    assert any(
        "Defaulting to replaying one source file from batch_002 in batch_004" in warning
        for warning in result["warnings"]
    )
    assert any(
        "Schema drift will occur in batch 3 by adding network_events.signal_strength as double"
        in warning
        for warning in result["warnings"]
    )


def test_semantically_incomplete_agent_schema_is_repaired(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: {
            "domain": "custom_schema",
            "name": "Bad Telecom Draft",
            "seed": 42,
            "table_specs": [
                {
                    "name": "event_sources",
                    "row_count": 5_000,
                    "columns": [
                        {
                            "name": "event_source_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        }
                    ],
                },
                {
                    "name": "events",
                    "row_count": 100_000,
                    "columns": [
                        {
                            "name": "event_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "event_source_id", "type": "long"},
                    ],
                },
            ],
            "relationships": [
                {
                    "name": "sources_events",
                    "parent_table": "event_sources",
                    "parent_column": "event_source_id",
                    "child_table": "events",
                    "child_column": "event_source_id",
                }
            ],
            "issues": [
                {
                    "type": "late_arrival",
                    "table": "incident_closures",
                    "column": "closure_ts",
                    "rate": 0.10,
                    "parameters": {
                        "semantics": "late_arriving_data",
                        "event_time_column": "closure_ts",
                        "arrival_column": "ingestion_ts",
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(
        scenario_service,
        "_repair_custom_schema_intent_with_model",
        lambda prompt, invalid_intent, error: {
            "domain": "custom_schema",
            "name": "Telecom Network Events",
            "seed": 42,
            "table_specs": [
                {
                    "name": "customer_regions",
                    "row_count": 100,
                    "columns": [
                        {
                            "name": "region_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "region_code", "type": "string"},
                    ],
                },
                {
                    "name": "cell_towers",
                    "row_count": 2_000,
                    "columns": [
                        {
                            "name": "tower_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "region_id", "type": "long"},
                    ],
                },
                {
                    "name": "network_engineers",
                    "row_count": 500,
                    "columns": [
                        {
                            "name": "engineer_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        }
                    ],
                },
                {
                    "name": "network_events",
                    "row_count": 100_000,
                    "columns": [
                        {
                            "name": "event_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "tower_id", "type": "long"},
                    ],
                },
                {
                    "name": "incidents",
                    "row_count": 8_000,
                    "columns": [
                        {
                            "name": "incident_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "event_id", "type": "long"},
                        {"name": "assigned_engineer_id", "type": "long"},
                    ],
                },
                {
                    "name": "incident_closures",
                    "row_count": 8_000,
                    "columns": [
                        {
                            "name": "closure_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "incident_id", "type": "long"},
                        {"name": "closure_ts", "type": "timestamp"},
                    ],
                },
            ],
            "relationships": [
                {
                    "name": "towers_events",
                    "parent_table": "cell_towers",
                    "parent_column": "tower_id",
                    "child_table": "network_events",
                    "child_column": "tower_id",
                },
                {
                    "name": "events_incidents",
                    "parent_table": "network_events",
                    "parent_column": "event_id",
                    "child_table": "incidents",
                    "child_column": "event_id",
                },
                {
                    "name": "incidents_closures",
                    "parent_table": "incidents",
                    "parent_column": "incident_id",
                    "child_table": "incident_closures",
                    "child_column": "incident_id",
                },
            ],
            "issues": [
                {
                    "type": "referential_orphan",
                    "table": "network_events",
                    "column": "tower_id",
                    "rate": 0.01,
                    "parameters": {},
                },
                {
                    "type": "null_value",
                    "table": "incidents",
                    "column": "assigned_engineer_id",
                    "rate": 0.01,
                    "parameters": {},
                },
                {
                    "type": "late_arrival",
                    "table": "incident_closures",
                    "column": "closure_ts",
                    "rate": 0.01,
                    "parameters": {
                        "semantics": "late_arriving_data",
                        "event_time_column": "closure_ts",
                        "arrival_column": "ingestion_ts",
                    },
                },
                {
                    "type": "invalid_value",
                    "table": "customer_regions",
                    "column": "region_code",
                    "rate": 0.01,
                    "parameters": {"value": "REGION_@@"},
                },
            ],
        },
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build telecom network events with 100k events, orphan cell tower IDs, "
        "missing network engineers, late incident closures, and invalid customer regions."
    )

    assert result["tables"]["network_events"] == 100_000
    assert "cell_towers" in result["tables"]
    assert "network_engineers" in result["tables"]
    assert "incident_closures" in result["tables"]
    assert any("repair agent fixed it" in warning for warning in result["warnings"])


def test_retail_promotions_issue_mapping_uses_requested_nouns(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: {
            "domain": "custom_schema",
            "name": "Retail Promotions",
            "seed": 42,
            "table_specs": [
                {
                    "name": "promotions",
                    "row_count": 100_000,
                    "columns": [
                        {
                            "name": "promotion_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "promotion_name", "type": "string"},
                        {"name": "product_id", "type": "long"},
                        {"name": "store_id", "type": "long"},
                        {"name": "start_date", "type": "date"},
                        {"name": "end_date", "type": "date"},
                    ],
                },
                {
                    "name": "products",
                    "row_count": 20_000,
                    "columns": [
                        {
                            "name": "product_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "product_name", "type": "string"},
                    ],
                },
                {
                    "name": "stores",
                    "row_count": 500,
                    "columns": [
                        {
                            "name": "store_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "store_name", "type": "string"},
                    ],
                },
                {
                    "name": "customers",
                    "row_count": 200_000,
                    "columns": [
                        {
                            "name": "customer_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "postal_code", "type": "string"},
                    ],
                },
                {
                    "name": "coupons",
                    "row_count": 200_000,
                    "columns": [
                        {
                            "name": "coupon_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "promotion_id", "type": "long"},
                        {"name": "product_id", "type": "long"},
                        {"name": "store_id", "type": "long"},
                        {"name": "issue_date", "type": "date"},
                    ],
                },
                {
                    "name": "coupon_redemptions",
                    "row_count": 150_000,
                    "columns": [
                        {
                            "name": "redemption_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "coupon_id", "type": "long"},
                        {"name": "customer_id", "type": "long"},
                        {"name": "redemption_ts", "type": "timestamp"},
                    ],
                },
            ],
            "relationships": [
                {
                    "name": "products_promotions",
                    "parent_table": "products",
                    "parent_column": "product_id",
                    "child_table": "promotions",
                    "child_column": "product_id",
                },
                {
                    "name": "stores_promotions",
                    "parent_table": "stores",
                    "parent_column": "store_id",
                    "child_table": "promotions",
                    "child_column": "store_id",
                },
                {
                    "name": "promotions_coupons",
                    "parent_table": "promotions",
                    "parent_column": "promotion_id",
                    "child_table": "coupons",
                    "child_column": "promotion_id",
                },
                {
                    "name": "coupons_coupon_redemptions",
                    "parent_table": "coupons",
                    "parent_column": "coupon_id",
                    "child_table": "coupon_redemptions",
                    "child_column": "coupon_id",
                },
                {
                    "name": "customers_coupon_redemptions",
                    "parent_table": "customers",
                    "parent_column": "customer_id",
                    "child_table": "coupon_redemptions",
                    "child_column": "customer_id",
                },
            ],
            "issues": [],
        },
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build retail promotions with 100k promotions, duplicate promotions, "
        "2% orphan product IDs, 5% missing store IDs, "
        "10% late coupon redemptions, schema drift, replayed files, "
        "and invalid customer postal codes."
    )

    assert result["tables"]["promotions"] == 100_000
    issue_by_type = {issue["type"]: issue for issue in result["issues"]}
    issue_targets = {(issue["type"], issue["table"], issue["column"]) for issue in result["issues"]}
    assert ("duplicate_record", "promotions", None) in issue_targets
    assert ("referential_orphan", "promotions", "product_id") in issue_targets
    assert ("null_value", "promotions", "store_id") in issue_targets
    assert ("late_arrival", "coupon_redemptions", "redemption_ts") in issue_targets
    assert ("invalid_value", "customers", "postal_code") in issue_targets
    assert issue_by_type["referential_orphan"]["rate"] == 0.02
    assert issue_by_type["null_value"]["rate"] == 0.05
    assert issue_by_type["late_arrival"]["rate"] == 0.1


def test_retail_promotions_adds_requested_fk_columns_to_primary_table(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: {
            "domain": "custom_schema",
            "name": "Retail Promotions",
            "seed": 42,
            "table_specs": [
                {
                    "name": "promotions",
                    "row_count": 100_000,
                    "columns": [
                        {
                            "name": "promotion_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "promotion_name", "type": "string"},
                    ],
                },
                {
                    "name": "products",
                    "row_count": 10_000,
                    "columns": [
                        {
                            "name": "product_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        }
                    ],
                },
                {
                    "name": "stores",
                    "row_count": 500,
                    "columns": [
                        {
                            "name": "store_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        }
                    ],
                },
                {
                    "name": "customers",
                    "row_count": 200_000,
                    "columns": [
                        {
                            "name": "customer_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "postal_code", "type": "string"},
                    ],
                },
                {
                    "name": "coupon_redemptions",
                    "row_count": 150_000,
                    "columns": [
                        {
                            "name": "redemption_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "redemption_ts", "type": "timestamp"},
                    ],
                },
            ],
            "relationships": [],
            "issues": [],
        },
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build retail promotions with 100k promotions, duplicate promotions, "
        "2% orphan product IDs, 5% missing store IDs, "
        "10% late coupon redemptions, schema drift, replayed files, "
        "and invalid customer postal codes."
    )

    promotion_columns = {column["name"] for column in result["columns"]["promotions"]}
    issue_targets = {(issue["type"], issue["table"], issue["column"]) for issue in result["issues"]}
    assert {"product_id", "store_id"}.issubset(promotion_columns)
    assert ("referential_orphan", "promotions", "product_id") in issue_targets
    assert ("null_value", "promotions", "store_id") in issue_targets


def test_rich_retail_sales_prompt_uses_agent_custom_schema_without_keyword_remap(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: {
            "domain": "custom_schema",
            "name": "North East Retail Sales",
            "seed": 42,
            "locale": "en-US",
            "timeline": {
                "start_date": "2025-01-01",
                "batches": 12,
                "frequency": "monthly",
            },
            "metadata": {
                "business_rules": [
                    "Sales only occur in North-East US states.",
                    "Returns only exist for delivered orders.",
                    "ship_date must be on or after order_date.",
                ],
                "statistical_anchors": {
                    "state_population_weights": {
                        "NY": 19_500_000,
                        "PA": 13_000_000,
                        "NJ": 9_300_000,
                        "MA": 7_000_000,
                        "CT": 3_600_000,
                        "ME": 1_400_000,
                        "NH": 1_400_000,
                        "RI": 1_100_000,
                        "VT": 650_000,
                    },
                    "order_amount_distribution": {
                        "type": "log_normal",
                        "median": 85,
                        "long_tail_max": 5000,
                    },
                    "channel_mix": {"online": 0.65, "in_store": 0.35},
                    "seasonality": {"nov_dec_lift": 0.40},
                },
            },
            "table_specs": [
                {
                    "name": "customers",
                    "row_count": 150_000,
                    "columns": [
                        {
                            "name": "customer_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {
                            "name": "state",
                            "type": "string",
                            "values": ["NY", "PA", "NJ", "MA", "CT", "ME", "NH", "RI", "VT"],
                        },
                        {"name": "city", "type": "string"},
                        {"name": "segment", "type": "string"},
                    ],
                },
                {
                    "name": "orders",
                    "row_count": 500_000,
                    "columns": [
                        {
                            "name": "order_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "customer_id", "type": "long", "nullable": False},
                        {"name": "order_date", "type": "date"},
                        {"name": "ship_date", "type": "date"},
                        {"name": "amount", "type": "decimal"},
                        {"name": "channel", "type": "string", "values": ["online", "in_store"]},
                        {"name": "record_status", "type": "string", "values": ["delivered"]},
                    ],
                },
                {
                    "name": "returns",
                    "row_count": 40_000,
                    "columns": [
                        {
                            "name": "return_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "order_id", "type": "long", "nullable": False},
                        {"name": "reason", "type": "string"},
                        {"name": "refund_amount", "type": "decimal"},
                    ],
                },
            ],
            "relationships": [
                {
                    "name": "customers_orders",
                    "parent_table": "customers",
                    "parent_column": "customer_id",
                    "child_table": "orders",
                    "child_column": "customer_id",
                },
                {
                    "name": "orders_returns",
                    "parent_table": "orders",
                    "parent_column": "order_id",
                    "child_table": "returns",
                    "child_column": "order_id",
                },
            ],
            "issues": [
                {
                    "type": "duplicate_record",
                    "table": "orders",
                    "column": None,
                    "rate": 0.01,
                    "parameters": {},
                },
                {
                    "type": "referential_orphan",
                    "table": "orders",
                    "column": "customer_id",
                    "rate": 0.02,
                    "parameters": {},
                },
                {
                    "type": "date_rule_violation",
                    "table": "orders",
                    "column": "ship_date",
                    "rate": 0.005,
                    "parameters": {"after_column": "order_date", "days_after": -1},
                },
                {
                    "type": "null_value",
                    "table": "returns",
                    "column": "reason",
                    "rate": 0.03,
                    "parameters": {},
                },
            ],
        },
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Generate 500,000 records for a retail sales scenario. Business rules: "
        "Sales only occur in North-East US states. Order volume per state is "
        "proportional to its population. ship_date must be on or after order_date. "
        "Returns only exist for delivered orders. Tables: customers, orders, returns. "
        "Relationships: one customer to many orders to some returns. Data issues: "
        "duplicate_record on orders 1%, referential_orphan on customer_id 2%, "
        "date_rule_violation ship before order 0.5%, null_value on returns.reason 3%. "
        "Settings: seed 42, timeline 2025-01-01 across 12 monthly batches, locale en-US, "
        "output Delta + raw."
    )

    assert result["domain"] == "custom_schema"
    assert result["tables"] == {"customers": 150_000, "orders": 500_000, "returns": 40_000}
    assert result["timeline"] == {
        "start_date": "2025-01-01",
        "batches": 12,
        "frequency": "monthly",
    }
    assert result["metadata"]["statistical_anchors"]["channel_mix"] == {
        "online": 0.65,
        "in_store": 0.35,
    }
    issue_targets = {(issue["type"], issue["table"], issue["column"]) for issue in result["issues"]}
    assert ("duplicate_record", "orders", None) in issue_targets
    assert ("referential_orphan", "orders", "customer_id") in issue_targets
    assert ("date_rule_violation", "orders", "ship_date") in issue_targets
    assert ("null_value", "returns", "reason") in issue_targets
    assert any("Schema-design agent designed" in warning for warning in result["warnings"])


def test_late_closure_adds_ingestion_column_for_late_arrival(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: {
            "domain": "custom_schema",
            "name": "Telecom Network Events",
            "seed": 42,
            "table_specs": [
                {
                    "name": "network_events",
                    "row_count": 100_000,
                    "columns": [
                        {
                            "name": "event_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        }
                    ],
                },
                {
                    "name": "incidents",
                    "row_count": 20_000,
                    "columns": [
                        {
                            "name": "incident_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "event_id", "type": "long"},
                    ],
                },
                {
                    "name": "incident_closures",
                    "row_count": 20_000,
                    "columns": [
                        {
                            "name": "closure_id",
                            "type": "long",
                            "primary_key": True,
                            "nullable": False,
                        },
                        {"name": "incident_id", "type": "long"},
                        {"name": "closure_ts", "type": "timestamp"},
                    ],
                },
            ],
            "relationships": [
                {
                    "name": "events_incidents",
                    "parent_table": "network_events",
                    "parent_column": "event_id",
                    "child_table": "incidents",
                    "child_column": "event_id",
                },
                {
                    "name": "incidents_closures",
                    "parent_table": "incidents",
                    "parent_column": "incident_id",
                    "child_table": "incident_closures",
                    "child_column": "incident_id",
                },
            ],
            "issues": [],
        },
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Build telecom network events with 100k events and 10% late incident closures."
    )

    closure_columns = {column["name"] for column in result["columns"]["incident_closures"]}
    late_issue = next(issue for issue in result["issues"] if issue["type"] == "late_arrival")
    assert "ingestion_ts" in closure_columns
    assert late_issue["column"] == "closure_ts"
    assert late_issue["parameters"]["event_time_column"] == "closure_ts"
    assert late_issue["parameters"]["arrival_column"] == "ingestion_ts"


def test_ai_model_ops_fallback_preserves_named_tables_and_issue_targets(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda prompt: None,
    )
    service = ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )

    result = service.create_scenario_from_prompt(
        "Create a synthetic AI model operations and evaluation dataset with 100,000 "
        "records representing a production AI platform used for model training, "
        "inference, monitoring, and feedback collection. Include users, models, "
        "prompts, responses, evaluations, and incident logs. Include 2% orphan "
        "model IDs, 2% orphan user IDs, 5% missing prompt categories, 5% missing "
        "evaluation labels, 5% missing inference latency values, 10% late-arriving "
        "feedback events, 10% duplicated inference records caused by replayed "
        "ingestion files, schema drift with renamed columns, extra fields, and "
        "inconsistent data types, invalid model versions, invalid tenant regions, "
        "feedback before inference, null or malformed response_text values, "
        "inconsistent status values, negative latency, future timestamps, and "
        "empty prompt text. Use tables such as user_events, prompt_requests, "
        "model_inferences, feedback_scores, evaluation_results, incident_logs, "
        "model_registry, and tenant_metadata."
    )

    assert result["domain"] == "custom_schema"
    assert result["tables"]["model_inferences"] == 100_000
    assert {
        "tenant_metadata",
        "user_directory",
        "user_events",
        "prompt_requests",
        "model_inferences",
        "feedback_scores",
        "evaluation_results",
        "incident_logs",
        "model_registry",
    }.issubset(set(result["tables"]))
    assert "representing" not in result["tables"]
    issue_targets = {
        (issue["type"], issue["table"], issue["column"]): issue for issue in result["issues"]
    }
    assert ("referential_orphan", "model_inferences", "model_id") in issue_targets
    assert ("referential_orphan", "model_inferences", "user_id") in issue_targets
    assert ("null_value", "prompt_requests", "prompt_category") in issue_targets
    assert ("null_value", "evaluation_results", "evaluation_label") in issue_targets
    assert ("null_value", "model_inferences", "response_latency_ms") in issue_targets
    assert ("late_arrival", "feedback_scores", "created_at") in issue_targets
    assert ("file_replay", "model_inferences", None) in issue_targets
    assert ("invalid_value", "model_inferences", "model_version") in issue_targets
    assert ("invalid_value", "tenant_metadata", "region") in issue_targets
    assert ("date_rule_violation", "feedback_scores", "created_at") in issue_targets
    assert ("invalid_format", "model_inferences", "response_text") in issue_targets
    assert ("blank_value", "prompt_requests", "prompt_text") in issue_targets
    assert issue_targets[("file_replay", "model_inferences", None)]["rate"] == 0.10
    assert issue_targets[("late_arrival", "feedback_scores", "created_at")]["parameters"][
        "arrival_column"
    ] == "ingestion_ts"
    drift = issue_targets[("schema_drift", "model_inferences", None)]
    assert drift["parameters"]["rename_columns"] == [
        {"from": "response_latency_ms", "to": "latency_ms", "batch": 4}
    ]
    assert drift["parameters"]["type_changes"] == [
        {"column": "confidence_score", "from": "decimal", "to": "string", "batch": 5}
    ]
    assert any("Invalid model-version and tenant-region rates" in w for w in result["warnings"])
