# Contributing

Keep Scenario Data Factory focused on Databricks-native synthetic data generation with deliberate
issues. Production logic belongs in importable Python modules under `src/scenario_data_factory`.

Before opening a change:

```bash
uv run pytest
uv run ruff check .
uv build
```

Do not commit secrets, personal access tokens, workspace-specific IDs, fabricated screenshots, or
real customer data.
