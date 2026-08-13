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
5. A Unity Catalog-enabled Databricks workspace with an existing catalog they can use. The bundle
   creates the schema and managed volumes inside that catalog.
6. Permission to create schemas, managed volumes, Jobs, MLflow experiments, and a Databricks App;
   and to create/write Delta tables in the generated schema.
7. A model-serving or Foundation Model endpoint the App service principal can query.

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

### First-Time Deployment In Your Own Workspace

Teammates do **not** need to manually create the schema or the `sdf_control` and `sdf_raw`
volumes. `databricks bundle deploy` provisions them. They do need an existing Unity Catalog
catalog and the permissions to create objects within it. A platform administrator should provide:

- a catalog name, such as `team_sandbox`;
- a queryable model-serving or Foundation Model endpoint;
- permission for the deployer to use the catalog and create schemas, volumes, Jobs, experiments,
  and Apps;
- permission for the deployed App service principal to query the model, manage generation-job
  runs, and write to the control/raw volumes if those grants are not applied automatically.

Before deployment, choose values that belong to their workspace:

```powershell
$Profile = "my-workspace-profile"
$Catalog = "team_sandbox"             # Existing Unity Catalog catalog
$Schema = "scenario_data_factory"     # Bundle creates this schema and its volumes
$Endpoint = "my-model-serving-endpoint"
$AppName = "scenario-data-factory-analytics"
```

Authenticate and verify the target workspace:

```powershell
databricks auth login --host https://<workspace-host>
databricks current-user me --profile $Profile
```

Build and deploy all workspace resources:

```powershell
uv build
databricks bundle validate --profile $Profile --target dev `
  --var catalog=$Catalog --var schema=$Schema `
  --var app_name=$AppName --var model_endpoint=$Endpoint
databricks bundle deploy --profile $Profile --target dev `
  --var catalog=$Catalog --var schema=$Schema `
  --var app_name=$AppName --var model_endpoint=$Endpoint
```

Then deploy the App using the source path returned by the bundle:

```powershell
$Summary = databricks bundle summary --profile $Profile --target dev `
  --var catalog=$Catalog --var schema=$Schema `
  --var app_name=$AppName --var model_endpoint=$Endpoint -o json | ConvertFrom-Json
$SourcePath = $Summary.resources.apps.scenario_data_factory.source_code_path

databricks apps start $AppName --profile $Profile
databricks apps deploy $AppName --profile $Profile `
  --source-code-path $SourcePath --auto-approve --timeout 20m
databricks apps get $AppName --profile $Profile
```

The App resource bindings request access to the selected model endpoint, generation Job, MLflow
experiment, and the two volumes. If the deployment succeeds but drafting or job submission is
denied, an administrator must grant the deployed App service principal the requested permissions.

## Deploy To Databricks

For subsequent code updates, rerun the validate, bundle deploy, and App deploy commands from the
first-time deployment section using the same `$Profile`, `$Catalog`, `$Schema`, `$Endpoint`, and
`$AppName` values. The deployment updates the wheel, Jobs, volumes, and App configuration without
requiring manual recreation of those resources.

### Install Command

The repository also provides an installer for the same standard dev deployment. Supply the values
for the teammate's workspace instead of relying on the sample defaults:

```powershell
uv run sdf install <profile> --target dev `
  --catalog <existing-catalog> `
  --schema scenario_data_factory `
  --app-name scenario-data-factory-<team> `
  --model-endpoint <model-serving-endpoint>
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

Delta table names use `<scenario>_<entity>` for the clean baseline and
`<scenario>_<entity>_dq` for the defect-injected version. For example, a Canadian banking
scenario named `cbank` writes `cbank_customers` and `cbank_customers_dq`; run IDs, row counts,
and hashes stay in the manifest rather than the table name.

The agent supplies this compact scenario code as `dataset_code`; reviewed YAML/JSON specs can set
`outputs.delta_namespace` directly.

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
