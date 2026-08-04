from __future__ import annotations

from scenario_data_factory.issues.null_value import NullValuePlugin


class CorrelatedMissingnessPlugin(NullValuePlugin):
    issue_type = "correlated_missingness"
