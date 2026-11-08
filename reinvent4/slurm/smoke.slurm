#!/bin/bash
# Phase 0 smoke run: 300 oracle calls, parp1, seed 0.
#   sbatch reinvent4/slurm/smoke.slurm [extra runner args...]
#
# Self-contained on purpose: it starts the benchmark's QuickVina HTTP service on this same
# node (127.0.0.1) and points REINVENT4 at it, so there is no hostname coordination and no
# long-lived oracle job to clean up. The oracle and REINVENT4 live in different conda envs
# -- the reinvent4 env deliberately has no OpenBabel -- so each is invoked by its env's
# absolute python path rather than by activating twice.
#
# What this run is meant to prove, beyond "it does not crash":
#   history.csv must hold EXACTLY --max_oracle_calls data rows. That is the budget
#   invariant the unit tests pin against a fake sampler; here it meets the real one.
#SBATCH --job-name=reinvent4_smoke
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_smoke/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_smoke/%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/gsimonyan/PMO-Dock}"
export WEKA_ROOT="${WEKA_ROOT:-/mnt/weka/$USER}"
# Kept apart from $WEKA_ROOT/results so a smoke run never lands among benchmark numbers.
export OUT_DIR="${OUT_DIR:-$WEKA_ROOT/results-smoke}"
export REINVENT_PRIOR_BASE="${REINVENT_PRIOR_BASE:-$WEKA_ROOT/large-assets/reinvent4/src/priors}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/reinvent4/plugins:${PYTHONPATH:-}"

ORACLE_PY="${ORACLE_PY:-/home/gsimonyan/envs/vina-oracle/bin/python}"
REINVENT_PY="$WEKA_ROOT/envs/reinvent4/bin/python"
BUDGET="${BUDGET:-300}"
TARGET="${TARGET:-parp1}"

export NUM_CPU_DOCK="${NUM_CPU_DOCK:-2}"
export MAX_CONCURRENT_DOCKING="${MAX_CONCURRENT_DOCKING:-$(( ${SLURM_CPUS_PER_TASK:-32} / NUM_CPU_DOCK ))}"
# obabel is resolved off PATH by benchmark/docking_oracle/docking.py.
export PATH="$(dirname "$ORACLE_PY"):$PATH"

echo "host:    $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "oracle:  $ORACLE_PY"
echo "agent:   $REINVENT_PY"
echo "out:     $OUT_DIR"
echo "budget:  $BUDGET   target: $TARGET"
echo "docking: num_cpu=$NUM_CPU_DOCK max_concurrent=$MAX_CONCURRENT_DOCKING"

# --- start the docking service on a free local port -------------------------------------
PORT=5050
while ss -tln 2>/dev/null | grep -q ":${PORT} "; do
  PORT=$((PORT + 1))
done
ORACLE_URL="http://127.0.0.1:${PORT}"
ORACLE_LOG="/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_smoke/${SLURM_JOB_ID:-local}.oracle.log"

cd "$PROJECT_ROOT"
"$ORACLE_PY" benchmark/docking_oracle/oracle_app.py --host 127.0.0.1 --port "$PORT" \
  > "$ORACLE_LOG" 2>&1 &
ORACLE_PID=$!
trap 'kill "$ORACLE_PID" 2>/dev/null || true' EXIT

echo "oracle pid $ORACLE_PID on $ORACLE_URL, log $ORACLE_LOG"
for _ in $(seq 1 120); do
  if curl -fsS "$ORACLE_URL/health" > /dev/null 2>&1; then break; fi
  if ! kill -0 "$ORACLE_PID" 2>/dev/null; then
    echo "FAIL: oracle died during startup; see $ORACLE_LOG"; tail -40 "$ORACLE_LOG"; exit 1
  fi
  sleep 2
done
curl -fsS "$ORACLE_URL/health" || { echo "FAIL: oracle not healthy"; tail -40 "$ORACLE_LOG"; exit 1; }
echo

# --- run REINVENT4 ----------------------------------------------------------------------
export DOCKING_VINA_URL="$ORACLE_URL"
START=$(date +%s)
# Deliberately not under `set -e`: a non-zero exit is itself a smoke-test result, and the
# history.csv diagnostics below are most interesting exactly when the run did not finish.
set +e
"$REINVENT_PY" reinvent4/reinvent_hit_runner.py \
  --config_file reinvent4/configs/hit.json \
  --oracle_name "$TARGET" \
  --seeds 0 \
  --max_oracle_calls "$BUDGET" \
  --reward_type geam \
  --oracle_url "$ORACLE_URL" \
  --max_workers 1 \
  "$@"
RC=$?
set -e
ELAPSED=$(( $(date +%s) - START ))
echo "runner exit=$RC  wall=${ELAPSED}s"

# --- verify the budget invariant --------------------------------------------------------
HIST=$(find "$OUT_DIR/reinvent4" -name history.csv -printf '%T@ %p\n' 2>/dev/null \
         | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$HIST" ]]; then
  echo "FAIL: no history.csv written under $OUT_DIR/reinvent4"; exit 1
fi
ROWS=$(( $(wc -l < "$HIST") - 1 ))
echo "history: $HIST"
echo "header:  $(head -1 "$HIST")"
echo "rows:    $ROWS   (expected $BUDGET)"
echo "distinct canonical smiles: $(tail -n +2 "$HIST" | cut -d, -f2 | sort -u | wc -l)"
ls -la "$(dirname "$HIST")"

if [[ "$ROWS" -ne "$BUDGET" ]]; then
  echo "FAIL: budget invariant violated ($ROWS != $BUDGET)"; exit 1
fi
if [[ "$RC" -ne 0 ]]; then
  echo "FAIL: runner exited $RC"; exit "$RC"
fi
echo "PASS: budget invariant holds end to end, runner exited 0."
