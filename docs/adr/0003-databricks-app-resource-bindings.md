# ADR 0003: Databricks App Resource Bindings

## Status

Accepted.

## Decision

The Databricks App reads model endpoint, Lakeflow job ID, MLflow experiment ID, and UC volume paths
from app resource bindings exposed as environment variables.

## Consequences

The app avoids hardcoded workspace IDs and personal tokens.
