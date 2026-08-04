# ADR 0002: Agent Confirmation Boundary

## Status

Accepted.

## Decision

The agent may create, patch, validate, estimate, and prepare scenarios. It does not get a direct
tool to submit full generation. Generation requires a deterministic spec hash confirmation.

## Consequences

Conversational ambiguity cannot start expensive or destructive workspace runs.
