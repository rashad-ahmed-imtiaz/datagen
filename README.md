# Scenario Data Factory

Scenario Data Factory is a Databricks-native synthetic data generator. Describe the business
domain, tables, rules, distributions, timeline, and intentional data defects in plain language;
the App's model-backed schema-design agent produces an executable scenario. The Spark generation
engine then creates relationally consistent synthetic data with `dbldatagen`, injects the approved
defects, and records a manifest of what changed.

The project is deliberately split into two layers:

- **Authoring:** the Databricks App uses a bound model-serving endpoint to infer tables, columns,
  relationships, distributions, business rules, batches, and data-quality issues from a prompt.
- **Execution:** an approved YAML or JSON `ScenarioSpec` is validated and generated without an
  agent. This keeps production generation reproducible and reviewable.

No model-generated Python or SQL is executed. The model produces a typed scenario contract; the
engine validates it before any generation job can run.

## What You Can Build

Use a natural-language request for any business domain, for example:

```text
Generate 500,000 records for a retail sales scenario with customers, orders, and returns.
Sales are limited to North-East US states and order volume is population weighted. Orders have a
65/35 online-to-store split and a November/December seasonal lift. Ship dates must be on or after
order dates, and returns only exist for delivered orders. Create 12 monthly batches starting
2025-01-01. Inject 1% duplicate orders, 2% orphan customer IDs, 0.5% ship-before-order records,
and 3% missing return reasons. Use seed 42 and write Delta plus raw batches.
```

The App turns that request into tables, semantic columns, relationships, distributions, batch
contracts, issue rules, and an auditable preview. The generated values are intended to look like
the domain, rather than placeholder sequences such as `city_1` or `name_1`.

## Features

- Uses Databricks Labs `dbldatagen` for baseline Spark DataFrames.
- Uses a model endpoint for natural-language scenario design; no fixed business-domain blueprint is
  selected for a prompt.
- Supports an agent-independent YAML/JSON execution path for reviewed scenarios.
- Generates clean Delta tables, dirty Delta tables, raw ingestion batches, issue manifests, and run
  summaries.
- Supports referential relationships, date constraints, parent-child filters, aggregate caps,
  weighted distributions, time-based batches, and realistic semantic values.
- Supports deliberate defects including nulls, duplicates, invalid values, referential orphans,
  date-rule violations, late arrival, file replay, schema drift, and correlated missingness.
- Requires a hash confirmation before a Databricks generation job is submitted.

## Prerequisites

You need:

1. Git.
2. Python 3.10 or later (Python 3.11 is recommended).
3. [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python environment and
   dependency management.
4. The [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install).
5. A Databricks workspace where you can create a schema, volumes, jobs, an experiment, and a
   Databricks App.
6. A model-serving or Foundation Model endpoint the App service principal can query.

For local App authoring, the same endpoint must be reachable through your authenticated Databricks
CLI profile. Full-scale data generation should run in the deployed Databricks job.

## Download And Install

These commands use PowerShell on Windows. The same `uv` and `databricks` commands work in other
shells with their normal environment-variable syntax.

```powershell
git clone https://github.com/rashad-ahmed-imtiaz/datagen.git
Set-Location datagen
uv sync --extra app --extra spark --extra dev
```

Check that the package and baseline generator are available:

```powershell
uv run sdf doctor
uv run sdf issues
```

## Configure Databricks Authentication

Log in to the target workspace, then verify the named profile. Replace the hostname and profile
name with your own values.

```powershell
databricks auth login --host https://<your-workspace-host>
databricks auth profiles
databricks current-user me --profile <profile>
```

The bundle defaults are `catalog=sdf`, `schema=scenario_data_factory`, and
`app_name=scenario-data-factory`. Override them during deployment when your workspace uses
different names.

## Deploy To Databricks

Set the profile and endpoint once for the current PowerShell session:

```powershell
$Profile = "<profile>"
$Endpoint = "<model-serving-endpoint>"
```

Validate and deploy the bundle. This builds the wheel, creates the schema and volumes, and deploys
the initialize, preview, generate, and smoke-test jobs.

```powershell
uv build
databricks bundle validate --profile $Profile --target dev --var model_endpoint=$Endpoint
databricks bundle deploy --profile $Profile --target dev --var model_endpoint=$Endpoint
```

Deploy the Databricks App from the source path resolved by the bundle:

```powershell
$Summary = databricks bundle summary --profile $Profile --target dev `
  --var model_endpoint=$Endpoint -o json | ConvertFrom-Json
$SourcePath = $Summary.resources.apps.scenario_data_factory.source_code_path

databricks apps start scenario-data-factory --profile $Profile
databricks apps deploy scenario-data-factory --profile $Profile `
  --source-code-path $SourcePath --auto-approve --timeout 20m
databricks apps get scenario-data-factory --profile $Profile
```

The App has resource bindings for the model endpoint, generation job, MLflow experiment, control
volume, and raw-data volume. Grant the App service principal access to the selected catalog if your
workspace does not grant the required permissions automatically.

### Install Command

The repository also provides an installer that performs the standard dev deployment using the
default catalog and schema:

```powershell
uv run sdf install <profile> --target dev --model-endpoint <model-serving-endpoint>
```

Use `--engine-only` to deploy the jobs and storage resources without deploying the App.

## Run The App Locally

Local App use still requires Databricks authentication and a model endpoint. Configure the
endpoint and local output locations, then run the FastAPI server:

```powershell
$env:DATABRICKS_CONFIG_PROFILE = "<profile>"
$env:SDF_MODEL_ENDPOINT = "<model-serving-endpoint>"
$env:SDF_CONTROL_VOLUME = "$PWD\.sdf\control"
$env:SDF_RAW_RUNS_ROOT = "$PWD\.sdf\raw\runs"

uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Enter a natural-language scenario request,
select **Draft**, inspect the inferred tables, relationships, rules, and defects, then select
**Prepare Run**. Local mode supports drafting and previewing. To submit a generation job, use the
deployed App resource binding or set `SDF_GENERATION_JOB_ID` to a job the authenticated principal
can run. Confirm the displayed spec hash before submitting the job.

`app/start_server.py` is the Databricks App entry point. It uses the MLflow Agent Server runtime,
so use the `uvicorn` command above for the lightweight local API/UI server.

## Run A Reviewed YAML Or JSON Scenario

The command-line path does not need a model endpoint. It validates and executes an existing spec.

```powershell
uv run sdf validate examples\insurance_claims.yaml
uv run sdf estimate examples\insurance_claims.yaml
uv run sdf generate examples\insurance_claims.yaml --output-root .\out
```

`generate` starts a local Spark session, so it requires the `spark` extra installed above. For
production-scale runs, use the Databricks generation job instead:

```powershell
databricks bundle run generate --profile <profile> --target dev `
  --var model_endpoint=<model-serving-endpoint> `
  --params "scenario_path=/Volumes/<catalog>/<schema>/sdf_control/scenarios/current.yaml,output_root=/Volumes/<catalog>/<schema>/sdf_raw/runs"
```

To submit from the App, do not bypass the confirmation API: the displayed spec hash is the required
approval guard.

## Agent Prompting Guide

The agent has the strongest execution contract when a request includes:

- the domain and approximate row count;
- required tables, key columns, and cardinalities when they are known;
- business rules as explicit constraints;
- distributions or statistical anchors, including their target table and column;
- every intentional issue with a table/column, rate or count, and semantics;
- timeline, batch count, locale, seed, and required outputs.

For example, say `late_arrival on payments.ingestion_ts: 5%, delayed 1-7 days`, rather than only
`late payments`. For low-volume reference tables, prefer an exact count, such as
`one invalid customer_regions.region_code record`, instead of a percentage that can round to zero.
For schema drift and file replay, request executable details: batch, mutation, source batch, and
target batch.

The planner refuses to submit an incomplete or contradictory contract. When that happens, no tables
or data have been generated; clarify the request and draft again. The execution engine, however,
remains available for a reviewed YAML/JSON spec regardless of model availability.

## Outputs And Reproducibility

Delta output is used for typed clean and dirty tables. Raw output represents physical ingestion
conditions such as schema-varying files, malformed records, ordering, late batches, and replay.
Each run produces a summary and manifest containing the issue ID, type, table, column, affected
record key, batch, and details.

The scenario has canonical JSON and a SHA-256 spec hash. Issue selection derives from the scenario
seed, scenario ID, issue ID, issue type, and record key so a reviewed spec can be reproduced.

## Supported Issues

- `null_value`, `blank_value`, `invalid_format`, `invalid_value`
- `duplicate_record`, `referential_orphan`, `date_rule_violation`
- `late_arrival`, `out_of_order`, `file_replay`, `schema_drift`
- `correlated_missingness`

## Development

Run the formatter/linter check and build a wheel:

```powershell
uv run ruff check .
uv build
```

The test suite is available as `uv run pytest` for contributors who want to run it.

## Security And Scope

All generated data is synthetic. Do not provide real customer data in a prompt or commit tokens,
workspace IDs, or generated production-like data. The v1 architecture intentionally does not depend
on Supervisor API, MCP, Genie, Lakebase, Vector Search, Unity AI Gateway, or external SaaS services.
