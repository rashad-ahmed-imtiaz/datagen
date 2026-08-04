from __future__ import annotations

from scenario_data_factory.issues.base import MetadataOnlyIssuePlugin


class SchemaDriftPlugin(MetadataOnlyIssuePlugin):
    issue_type = "schema_drift"
    requires_raw_output = True
