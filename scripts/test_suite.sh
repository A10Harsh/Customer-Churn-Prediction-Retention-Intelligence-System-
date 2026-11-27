#!/usr/bin/env bash
# Keep native numerical runtimes from oversubscribing threads during pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=2
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$(mktemp -d)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
"${PYTHON_BIN:-.venv/bin/python}" -m pytest \
  --cov=retention --cov-branch --cov-report=term-missing \
  --durations=20 -o faulthandler_timeout=60 "$@"
