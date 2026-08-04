from __future__ import annotations

from scenario_data_factory.issues.base import MetadataOnlyIssuePlugin


class FileReplayPlugin(MetadataOnlyIssuePlugin):
    issue_type = "file_replay"
    requires_raw_output = True
