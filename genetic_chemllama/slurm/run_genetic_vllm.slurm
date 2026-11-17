#!/bin/bash
# genetic_chemllama with generation served by a shared vLLM server.
#
# The search loop, oracle, prompts, budget and configs are IDENTICAL to the
# ordinary launchers. Only the generation backend differs:
#
#   hf   (default): N seed processes, N copies of the weights, HF generate,
#                   all time-slicing one GPU.
#   vllm (here):    one server holds one copy of the weights and batches
#                   continuously across every seed process.
#
# Nothing about this changes what the oracle sees. The backend returns
# prompt + completion exactly as the HF path does, and stops at the first
# [/SMILES] -- which find_valid_mols never reads past -- so the extracted
# molecule is unchanged while the tokens after it are never generated.
#
# Usage:
#   sbatch --export=ALL,TASK_NAME=hit.parp1 genetic_chemllama/slurm/run_genetic_vllm.slurm
#
#   TASK_NAME     required, e.g. hit.parp1 / lead.jak2_06_0 / spec.6nzp_7uyt
#   CONFIG        base config (default: genetic/configs/llama3_1b.yaml)
#   SEEDS         default "1 2 3 4 5"
#   MAX_CALLS     default 3000
#   REWARD_TYPE   default hit
#   MODEL_ID      default taken from the config's checkpoint_path
#   VINA_URL      optional docking service URL
#
# The vLLM server runs in its OWN conda env (chemllama-vllm) and is reached
# over HTTP, so it cannot disturb the torch/transformers pins in mol-grpo.
#
#SBATCH --job-name=genetic_vllm
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/genetic_vllm/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/genetic_vllm/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_NAME="${TASK_NAME:?Set TASK_NAME, e.g. --export=ALL,TASK_NAME=hit.parp1}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/genetic_chemllama/genetic/configs/llama3_1b.yaml}"
SEEDS="${SEEDS:-1 2 3 4 5}"
MAX_CALLS="${MAX_CALLS:-3000}"
REWARD_TYPE="${REWARD_TYPE:-hit}"
MAX_WORKERS="${MAX_WORKERS:-15}"

CHEM_ENV="${CHEM_ENV:-mol-grpo}"
VLLM_ENV="${VLLM_ENV:-chemllama-vllm}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/weka/shared-cache/miniforge3}"
VLLM_PYTHON="${VLLM_PYTHON:-${CONDA_ROOT}/envs/${VLLM_ENV}/bin/python}"

# Port derived from the job id so two jobs on one node cannot collide.
VLLM_PORT="${VLLM_PORT:-$(( 8000 + (${SLURM_JOB_ID:-0} % 2000) ))}"
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_GPU_FRAC="${VLLM_GPU_FRAC:-0.85}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-2048}"

mkdir -p "/mnt/weka/${USER}/results/slurm_logs/genetic_vllm"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CHEM_ENV}"

test -x "${VLLM_PYTHON}" || { echo "ERROR: VLLM_PYTHON not executable: ${VLLM_PYTHON}" >&2; exit 1; }
test -f "${CONFIG}" || { echo "ERROR: no such config: ${CONFIG}" >&2; exit 1; }

# Model id: explicit, else whatever the config already points at, so this
# launcher never silently serves a different checkpoint than the run expects.
MODEL_ID="${MODEL_ID:-$(python - "$CONFIG" << 'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["model"]["checkpoint_path"])
PY
)}"

# This cluster has no system CUDA (/usr/local/cuda does not exist), so
# FlashInfer's JIT cannot find nvcc and the engine dies during startup
# profiling. Point CUDA_HOME at the toolkit pip installed inside the vLLM env,
# and leave the FlashInfer sampler off so startup never triggers a long JIT
# build.
VLLM_SITE="$("${VLLM_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
export CUDA_HOME="${CUDA_HOME:-${VLLM_SITE}/nvidia/cu13}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "GENETIC_VLLM: task=${TASK_NAME} config=${CONFIG}"
echo "GENETIC_VLLM: model=${MODEL_ID} port=${VLLM_PORT} gpu_frac=${VLLM_GPU_FRAC}"
echo "GENETIC_VLLM: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is empty" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Start the shared vLLM server
# ---------------------------------------------------------------------------
SERVER_LOG="/mnt/weka/${USER}/results/slurm_logs/genetic_vllm/${SLURM_JOB_ID:-0}.vllm.log"
echo "GENETIC_VLLM: starting vLLM server, log=${SERVER_LOG}"
"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --served-model-name "${MODEL_ID}" \
  --port "${VLLM_PORT}" \
  --host 127.0.0.1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "${VLLM_GPU_FRAC}" \
  --max-model-len "${VLLM_MAX_LEN}" \
  > "${SERVER_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "GENETIC_VLLM: stopping vLLM server (pid ${VLLM_PID})"
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "GENETIC_VLLM: waiting for server readiness"
PYTHONPATH="${PROJECT_ROOT}/genetic_chemllama${PYTHONPATH:+:${PYTHONPATH}}" python - << PY
from genetic.generation_backends import VLLMServerBackend
VLLMServerBackend.wait_until_ready("${VLLM_URL}", timeout_s=1800)
print("server ready", flush=True)
PY

# run.py reads these when the config does not set generation_backend*, and
# genetic_runner.py launches the per-seed processes with the environment it
# inherits -- so the stock configs are used completely unchanged.
export GENERATION_BACKEND=vllm
export VLLM_SERVICE_URL="${VLLM_URL}"

VINA_ARGS=()
if [[ -n "${VINA_URL:-}" ]]; then
  VINA_ARGS=(--vina_url "${VINA_URL}")
fi

python "${PROJECT_ROOT}/genetic_chemllama/genetic_runner.py" \
  --config_file "${CONFIG}" \
  --seeds ${SEEDS} \
  --task_name "${TASK_NAME}" \
  --max_oracle_calls "${MAX_CALLS}" \
  --reward_type "${REWARD_TYPE}" \
  --max_workers "${MAX_WORKERS}" \
  "${VINA_ARGS[@]}"
