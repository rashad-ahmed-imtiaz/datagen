# ADR 0001: dbldatagen as Baseline Generator

## Status

Accepted.

## Decision

Scenario Data Factory uses Databricks Labs `dbldatagen==0.4.0` as the required baseline Spark
DataFrame generator. Direct API usage is isolated in `DbldatagenEngine`.

## Consequences

Issue injection, manifests, app tools, and outputs remain independent of dbldatagen internals.
Spark generation must run where dbldatagen and compatible PySpark are available.
