from __future__ import annotations

from scenario_data_factory.issues.base import MetadataOnlyIssuePlugin


class OutOfOrderPlugin(MetadataOnlyIssuePlugin):
    issue_type = "out_of_order"
    requires_raw_output = True
