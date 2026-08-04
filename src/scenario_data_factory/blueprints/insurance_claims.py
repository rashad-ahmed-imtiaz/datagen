from __future__ import annotations

from datetime import date

from scenario_data_factory.blueprints.base import BlueprintMetadata, DomainBlueprint
from scenario_data_factory.models.scenario import (
    ColumnSpec,
    ColumnType,
    IssueSpec,
    IssueType,
    OutputMode,
    OutputSpec,
    RelationshipSpec,
    ScenarioSpec,
    TableSpec,
    TimelineSpec,
)


class InsuranceClaimsBlueprint(DomainBlueprint):
    metadata = BlueprintMetadata(
        domain="insurance_claims",
        name="Insurance claims",
        description="Customers, policies, claims, and payments with Guidewire-style claims flows.",
        tables=["customers", "policies", "claims", "payments"],
    )

    def build(self, *, name: str, seed: int, scale: str = "demo") -> ScenarioSpec:
        sizes = {
            "small": (100, 150, 300, 400),
            "demo": (10_000, 15_000, 30_000, 40_000),
        }
        customers, policies, claims, payments = sizes.get(scale, sizes["demo"])
        duplicate_claims = round(claims * 0.03)
        claim_orphans = round(claims * 0.01)
        missing_adjusters = round(claims * 0.05)
        late_payments = round(payments * 0.04)
        date_violations = round(claims * 0.02)
        replayed_claims = round((claims / 30) * 0.25)
        correlated_missing = round(claims * 0.02)
        invalid_customer_provinces = max(1, round(customers * 0.01))
        missing_customer_last_names = max(1, round(customers * 0.015))
        policy_date_violations = max(1, round(policies * 0.01))
        return ScenarioSpec(
            name=name,
            domain=self.metadata.domain,
            seed=seed,
            locale="en_CA",
            timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
            tables=[
                TableSpec(
                    name="customers",
                    row_count=customers,
                    source_systems=["guidewire", "partner_portal", "legacy_batch"],
                    columns=[
                        ColumnSpec(name="customer_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="first_name", type=ColumnType.STRING, faker="first_name"),
                        ColumnSpec(name="last_name", type=ColumnType.STRING, faker="last_name"),
                        ColumnSpec(
                            name="province", type=ColumnType.STRING, values=["ON", "QC", "BC", "AB"]
                        ),
                        ColumnSpec(name="created_date", type=ColumnType.DATE),
                    ],
                ),
                TableSpec(
                    name="policies",
                    row_count=policies,
                    source_systems=["guidewire", "legacy_batch"],
                    columns=[
                        ColumnSpec(name="policy_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="customer_id", type=ColumnType.LONG, nullable=False),
                        ColumnSpec(
                            name="policy_type",
                            type=ColumnType.STRING,
                            values=["auto", "home", "tenant"],
                        ),
                        ColumnSpec(name="effective_date", type=ColumnType.DATE),
                        ColumnSpec(name="expiry_date", type=ColumnType.DATE),
                    ],
                ),
                TableSpec(
                    name="claims",
                    row_count=claims,
                    source_systems=["guidewire", "partner_portal", "legacy_batch"],
                    columns=[
                        ColumnSpec(name="claim_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="policy_id", type=ColumnType.LONG, nullable=False),
                        ColumnSpec(
                            name="source_system",
                            type=ColumnType.STRING,
                            values=["guidewire", "partner_portal", "legacy_batch"],
                        ),
                        ColumnSpec(name="adjuster_id", type=ColumnType.STRING),
                        ColumnSpec(name="loss_date", type=ColumnType.DATE),
                        ColumnSpec(name="settlement_date", type=ColumnType.DATE),
                        ColumnSpec(
                            name="claim_amount", type=ColumnType.DECIMAL, precision=12, scale=2
                        ),
                    ],
                ),
                TableSpec(
                    name="payments",
                    row_count=payments,
                    source_systems=["guidewire", "legacy_batch"],
                    columns=[
                        ColumnSpec(name="payment_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="claim_id", type=ColumnType.LONG, nullable=False),
                        ColumnSpec(name="payment_date", type=ColumnType.DATE),
                        ColumnSpec(name="ingestion_ts", type=ColumnType.TIMESTAMP),
                        ColumnSpec(name="amount", type=ColumnType.DECIMAL, precision=12, scale=2),
                    ],
                ),
            ],
            relationships=[
                RelationshipSpec(
                    name="customers_policies",
                    parent_table="customers",
                    parent_column="customer_id",
                    child_table="policies",
                    child_column="customer_id",
                ),
                RelationshipSpec(
                    name="policies_claims",
                    parent_table="policies",
                    parent_column="policy_id",
                    child_table="claims",
                    child_column="policy_id",
                ),
                RelationshipSpec(
                    name="claims_payments",
                    parent_table="claims",
                    parent_column="claim_id",
                    child_table="payments",
                    child_column="claim_id",
                ),
            ],
            issues=[
                IssueSpec(
                    issue_id="iss_invalid_customer_provinces",
                    type=IssueType.INVALID_VALUE,
                    table="customers",
                    column="province",
                    exact_count=invalid_customer_provinces,
                    parameters={"value": "ZZ"},
                ),
                IssueSpec(
                    issue_id="iss_missing_customer_last_names",
                    type=IssueType.NULL_VALUE,
                    table="customers",
                    column="last_name",
                    exact_count=missing_customer_last_names,
                ),
                IssueSpec(
                    issue_id="iss_policy_effective_after_expiry",
                    type=IssueType.DATE_RULE_VIOLATION,
                    table="policies",
                    column="effective_date",
                    exact_count=policy_date_violations,
                    parameters={"after_column": "expiry_date", "days_after": 1},
                ),
                IssueSpec(
                    issue_id="iss_duplicate_claims",
                    type=IssueType.DUPLICATE_RECORD,
                    table="claims",
                    exact_count=duplicate_claims,
                ),
                IssueSpec(
                    issue_id="iss_policy_orphans",
                    type=IssueType.REFERENTIAL_ORPHAN,
                    table="claims",
                    column="policy_id",
                    exact_count=claim_orphans,
                ),
                IssueSpec(
                    issue_id="iss_missing_adjusters",
                    type=IssueType.NULL_VALUE,
                    table="claims",
                    column="adjuster_id",
                    exact_count=missing_adjusters,
                ),
                IssueSpec(
                    issue_id="iss_late_payments",
                    type=IssueType.LATE_ARRIVAL,
                    table="payments",
                    column="ingestion_ts",
                    exact_count=late_payments,
                    parameters={"delay_days_min": 1, "delay_days_max": 5},
                ),
                IssueSpec(
                    issue_id="iss_invalid_loss_dates",
                    type=IssueType.DATE_RULE_VIOLATION,
                    table="claims",
                    column="loss_date",
                    exact_count=date_violations,
                    parameters={"after_column": "settlement_date"},
                ),
                IssueSpec(
                    issue_id="iss_file_replay_batch_10_12",
                    type=IssueType.FILE_REPLAY,
                    table="claims",
                    exact_count=replayed_claims,
                    parameters={"source_batch": 10, "target_batch": 12},
                ),
                IssueSpec(
                    issue_id="iss_schema_drift_fraud_score",
                    type=IssueType.SCHEMA_DRIFT,
                    table="claims",
                    exact_count=1,
                    parameters={
                        "activation_batch": 20,
                        "add_columns": [{"name": "fraud_score", "type": "decimal"}],
                    },
                ),
                IssueSpec(
                    issue_id="iss_legacy_adjuster_correlation",
                    type=IssueType.CORRELATED_MISSINGNESS,
                    table="claims",
                    column="adjuster_id",
                    exact_count=correlated_missing,
                    parameters={"where": {"source_system": "legacy_batch", "after_batch": 15}},
                ),
            ],
            outputs=OutputSpec(
                mode=OutputMode.BOTH,
                include_clean=True,
                catalog="sdf",
                schema_name="scenario_data_factory",
            ),
        )
