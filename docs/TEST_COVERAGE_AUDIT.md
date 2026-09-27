# Stellium Test Coverage Audit

This inventory maps the requested software-testing checklist to evidence in this repository. “Covered” means a maintained automated check exists for the named concern; it does not mean every production configuration, failure mode, or deployment has been exercised. “Partial” means tests cover only a local, synthetic, or narrow slice. “Missing” means no repeatable check was found. “N/A” means the capability is not currently part of this repository's product surface.

## Verified Local Baseline

- Latest measured suite: 142 collected, 139 passed, 3 live TigerGraph checks skipped by default.
- Local source statement coverage: 79.98% overall (`pytest --cov=src`); CI enforces a 75% regression floor. Coverage is not a quality score or proof that critical paths are covered.
- Highest coverage: models and guardrails (100%), ingestion (93%), embeddings and bitemporal logic (87–88%).
- RAG and GraphRAG retrieval paths are now directly tested at 95% and 87%, respectively.
- Evaluation runner coverage is 88%, API coverage 82%, LLM router coverage 78%, and graph client coverage 76% after focused contract tests.
- Lowest remaining core areas: live graph verifier (55%), graph mock (67%), and ingestion batch upsert (68%). Cache validation rejects wrong-dimension, boolean, and non-finite vectors, and ingestion rejects embedding response cardinality mismatches.
- Live checks in `tests/test_graph_live.py` require `TG_RUN_LIVE_TESTS=1`; the default suite does not prove live cluster readiness.

## Checklist Mapping

| Checklist area | Current evidence | Status | Next useful step |
| :--- | :--- | :---: | :--- |
| Linting, formatting, type checking | Ruff configured; workflow runs Ruff and `mypy src/ tests/` | Covered | Keep the local and CI checks aligned |
| Static application security (SAST) | `tests/test_sast_and_guardrails.py`; Bandit runs in CI | Covered / partial | Retain machine-readable reports and review suppressions |
| Software composition / dependency vulnerability scanning | `pip-audit` runs on pushes, pull requests, and daily schedule | Covered / partial | Review exceptions explicitly; scanner coverage is dependencies, not application code |
| License compliance | CI emits a locked-environment `pip-licenses` JSON inventory as a retained artifact; unresolved and proprietary labels require human review | Partial | Resolve metadata gaps, document approved exceptions, and establish a reviewed policy before enforcing an allowlist |
| Dead code elimination analysis | CI runs Vulture on production `src/` at 80% confidence; tests are excluded to avoid mock/decorator false positives | Covered / partial | Review new findings and expand only when the tool can resolve framework registrations accurately |
| Complexity auditing | Ruff McCabe C901 gate in `pyproject.toml`, capped at the current maximum complexity of 24 | Covered / partial | Refactor the 11 existing hotspots, then lower the ceiling in small steps |
| Architecture and layer isolation | `tests/test_architecture_layers.py` checks imports and package boundaries | Covered | Extend checks to runtime wiring and forbidden dependency directions |
| Unit, component, mock, property, regression tests | Domain, coprocessor, ingestion, bitemporal, mocks, Hypothesis, and parser tests across `tests/` | Covered / partial | Add edge cases where coverage is low; retain bug reproductions as regressions |
| Integration and system testing | FastAPI `TestClient`, graph mock, embedding and provider mocks; compare route runs all three pipelines with retrieved text and agentic trace | Covered / partial | Add a deterministic full-corpus ingest → retrieval → answer → trace scenario |
| End-to-end testing | No browser-to-live-backend-to-live-graph test found | Missing | Add separately gated deployment smoke/E2E using a disposable or controlled dataset |
| Smoke and sanity testing | `tests/test_smoke.py`; CI starts API and checks health/snapshot | Covered / partial | Make CI smoke assert meaningful response contracts, not only HTTP success |
| API and schema/contract testing | `tests/test_api.py`, `tests/test_contract_and_schema_drift.py`, OpenAPI checks | Covered / partial | Add consumer contracts for the actual dashboard client when one exists |
| Consumer-driven contracts | No consumer-owned contract suite found | Missing | Add when a maintained frontend/client is in this repository |
| UI, visual regression, snapshot, accessibility, localization, browser, mobile | No `frontend/` application found; snapshot endpoint returns static fixture data | Missing / N/A currently | Build the actual dashboard first, then add Playwright, visual, a11y, locale and browser checks |
| Load, stress, concurrency, latency, volume | `test_load_and_stress.py` uses synthetic chunks with 10 worker threads; `test_concurrency_latency.py` overlaps 20 mocked LLM requests | Partial | Repeat against all 22,016 chunks and live graph; record p50/p95/p99, throughput and RSS |
| Endurance, soak, spike, scalability, capacity planning | No long-duration or stepped-capacity run found | Missing | Add a separately invoked benchmark profile and define capacity/SLO targets from measurements |
| Benchmark testing | RRF/coprocessor microbenchmarks exist; no saved full 100/50 comparison result set | Partial | Run all three pipelines on both datasets and persist raw results plus aggregate report |
| Chaos and fault injection | `test_chaos_resilience.py` injects selected fake HTTP/graph failures | Partial | Add timeouts, malformed/partial responses, retry exhaustion, and live dependency fault drills |
| Failover | Provider retry/fallback unit coverage exists in `tests/test_llm.py` | Partial | Verify fresh-run model lock and end-to-end failover with deterministic provider stubs |
| Disaster recovery, backup/restore | No backup/restore procedure or test found | Missing | Define graph export/rebuild recovery and rehearse restore against a disposable graph |
| Canary, dark launch, feature flags, synthetic monitoring, post-deploy smoke | No deployment workflow or monitoring configuration found | Missing | Add only with a deployment target; report deployment health separately from unit CI |
| DAST and secret detection | `tests/test_dast_security.py`, `tests/test_sast_and_guardrails.py` cover selected payloads and redaction | Partial | Add an automated scanner against a running test app and secret scanning in CI |
| IAST, penetration testing, vulnerability scanning | No instrumented runtime or penetration exercise found; dependency audit is not app scanning | Missing / partial | Schedule dependency scanning now; perform scoped runtime testing before public deployment |
| Identity/access, compliance, encryption at rest/in transit | No authorization matrix, compliance control mapping, or encryption verification suite found | Missing / environment-owned | Document provider responsibilities; verify TLS validation and secrets/access policy for deployment |
| Database migration, schema validation, data integrity | Schema contracts and mock graph tests exist; no versioned migration runner or full corpus integrity audit | Partial | Add idempotent schema upgrade tests and post-ingestion vertex/edge/vector reconciliation |
| Backup/restore, infrastructure-as-code, configuration drift | No IaC or backup manifests found | Missing / N/A currently | Add when infrastructure is managed in-repo; make required environment config validated at startup |
| Network latency simulation | Fake response/fault tests exist, but no controlled latency/loss simulation | Partial | Use a transport stub with delay, jitter, disconnect, and retry-after scenarios |
| User acceptance testing | No documented judge/user acceptance script found | Missing | Prepare reproducible scenarios spanning simple, multi-hop, conflict, and out-of-corpus questions |
| Backward/forward compatibility and upgrade paths | Some DTO/schema drift tolerance tests; no supported-version matrix or upgrade rehearsal | Partial | Define compatibility policy and exercise old/new API and graph-schema fixtures |

## Priority Plan

1. **Metric correctness:** keep EM/F1, retrieval completeness, token accounting, and trace fields mathematically and semantically consistent. Token F1 was changed to multiset overlap; regression tests now cover duplicate tokens and empty inputs.
2. **Agentic trace truthfulness:** distinguish agent identities from tool names; record planner, tool, and synthesis latency/tokens separately; deterministic GSQL operations now report zero LLM tokens and orchestrator usage is recorded in separate `llm_calls` entries. Distinct specialist agents remain a design gap.
3. **Meaningful CI gates:** source and test typing, Bandit, scheduled `pip-audit`, a measured 75% coverage floor, deterministic benchmark smoke assertions, and retained test artifacts are now in CI; validate hosted runs and continue raising the floor with evidence.
4. **Retrieval quality and evaluation:** direct RAG/GraphRAG/API tests and benchmark JSONL contract tests are present; complete the three-pipeline 100-question public run and all 50 hidden questions under the required pipeline protocol, preserving raw JSONL and run metadata.
5. **Real system and UX verification:** replace the static graph snapshot/metrics fixture with live result data; once a UI is present, add browser, accessibility, and visual checks.
6. **Performance evidence:** benchmark the full corpus and live Savanna under repeatable conditions; publish latency distributions, concurrency, memory, token/neuron cost, accuracy, and completeness together.
7. **Complexity reduction:** C901 currently prevents functions exceeding the measured baseline of 24; prioritize the ReAct parser, conflict resolver, corpus parser, and benchmark runner, then ratchet the ceiling downward.

## Scope Notes

Tests for mobile platforms, browser compatibility, UI visual regression, canary releases, dark launches, IaC drift, and deployment monitoring cannot be meaningfully claimed until Stellium has the corresponding UI/deployment/infrastructure artifacts. They remain gaps in the full delivery lifecycle, rather than reasons to add empty test placeholders.
