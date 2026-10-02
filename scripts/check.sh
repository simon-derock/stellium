#!/usr/bin/env bash
# Run the same quality gate as CI: tests, lint, formatting, strict typing, dead code, security.
# Usage: scripts/check.sh            (exits non-zero and names every failing step)
set -uo pipefail
cd "$(dirname "$0")/.."

failed=()
run() {
  local name="$1"
  shift
  if ! "$@" >/tmp/stellium-check-"$name".log 2>&1; then
    failed+=("$name")
    echo "FAIL  $name  (log: /tmp/stellium-check-$name.log)"
  else
    echo "ok    $name"
  fi
}

# Wall-clock budgets in the latency tests scale up on busy machines, as they do in CI.
run pytest env STELLIUM_SLA_SCALE="${STELLIUM_SLA_SCALE:-3}" uv run pytest -q
run ruff uv run ruff check
run format uv run ruff format --check
run mypy uv run mypy src/ tests/
run vulture uv run vulture src/ --min-confidence 80
run bandit uv run bandit -q -r src/ -ll
if [[ -d web/node_modules ]]; then
  run web-types bash -c "cd web && npx tsc -b"
  run web-tests bash -c "cd web && npx vitest run"
fi

if ((${#failed[@]})); then
  echo "gate failed: ${failed[*]}"
  exit 1
fi
echo "gate passed"
