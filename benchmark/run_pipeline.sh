#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the full benchmark pipeline: Step 5 (agent trajectories) + Step 6-9 (eval).

Usage:
  bash run_pipeline.sh <candidate_model>

Examples:
  bash run_pipeline.sh moonshotai/kimi-k2.5
  bash run_pipeline.sh openai/gpt-oss-120b

Environment variables:
  OPENROUTER_API_KEY       (optional if set in .env)
  OPENROUTER_URL           (default: https://openrouter.ai/api/v1)
  RUN_ROOT                 (default: ../data/smoke_hero_step321_eval)
  FRESH_MODEL_DIR          (default: false; when true, clears model folder before run)
  BASE_PREPARED_FILE       (default: latest *_validated_prepared.jsonl under GPT5-2run hero_run_v1)
  BASE_ANSWER_KEY_FILE     (default: derived from BASE_PREPARED_FILE)
  MCP_SERVER_DIR           (default: ../mcp_servers/smithery_mcp_servers_0210)
  AGENT_MODE               (default: multi; single or multi)
  AGENT_TOOLS              (default: virtual; real or virtual)
  VIRTUAL_TOOL_MODEL       (default: same as candidate model)
  USER_MAX_TURNS           (default: 5)
  MAX_WORKERS              (default: 16)
  TIMEOUT_SECONDS          (default: 1800)
  EVAL_DIMENSIONS          (default: tool_call,workflow_completion,followup_quality,autonomy,grounding)
  EVAL_CONCURRENCY         (default: 40)
  EVAL_MAX_TOKENS          (default: 4096)
  START_IDX                (optional)
  BATCH_SIZE               (optional)
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

CANDIDATE_MODEL="${1:-}"
JUDGE_MODEL="openai/gpt-oss-120b"

if [ -z "${CANDIDATE_MODEL}" ]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Force a fixed eval judge model for consistency.
if [ -n "${2:-}" ] && [ "${2}" != "${JUDGE_MODEL}" ]; then
    echo "Warning: judge model is fixed to ${JUDGE_MODEL}; ignoring provided value: ${2}"
fi

# Resolve Python executable (override-able).
if [ -n "${PYTHON_BIN:-}" ]; then
    :
elif [ -x "/Users/smrstep/miniconda3/envs/toucan/bin/python" ]; then
    PYTHON_BIN="/Users/smrstep/miniconda3/envs/toucan/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: neither 'python' nor 'python3' is available in PATH."
    exit 1
fi

OPENROUTER_URL="${OPENROUTER_URL:-https://openrouter.ai/api/v1}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

# If env var is missing, try to load OPENROUTER_API_KEY from local .env.
if [ -z "${OPENROUTER_API_KEY}" ] && [ -f ".env" ]; then
    dotenv_key_line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY[[:space:]]*=' .env | tail -n 1 || true)"
    if [ -n "${dotenv_key_line}" ]; then
        OPENROUTER_API_KEY="$(printf '%s' "${dotenv_key_line}" | sed -E 's/^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY[[:space:]]*=[[:space:]]*//')"
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY%\"}"
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY#\"}"
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY%\'}"
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY#\'}"
    fi
fi

# Sanitize accidental header formatting issues.
OPENROUTER_API_KEY="$(printf '%s' "${OPENROUTER_API_KEY}" | tr -d '\r\n' | sed -E 's/^[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+//')"

if [ -z "${OPENROUTER_API_KEY}" ]; then
    echo "Error: OPENROUTER_API_KEY not found in env or .env."
    exit 1
fi

DEFAULT_BASE_PREPARED="$(ls -t ../data/GPT5-2run/Toucan/data/hero_run_v1/processed/*_validated_prepared.jsonl 2>/dev/null | head -n 1 || true)"
BASE_PREPARED_FILE="${BASE_PREPARED_FILE:-${DEFAULT_BASE_PREPARED}}"
if [ -z "${BASE_PREPARED_FILE}" ] || [ ! -f "${BASE_PREPARED_FILE}" ]; then
    echo "Error: BASE_PREPARED_FILE not found: ${BASE_PREPARED_FILE}"
    exit 1
fi

DEFAULT_BASE_ANSWER_KEY="${BASE_PREPARED_FILE%_validated_prepared.jsonl}_answer_key.jsonl"
BASE_ANSWER_KEY_FILE="${BASE_ANSWER_KEY_FILE:-${DEFAULT_BASE_ANSWER_KEY}}"
if [ ! -f "${BASE_ANSWER_KEY_FILE}" ]; then
    echo "Error: BASE_ANSWER_KEY_FILE not found: ${BASE_ANSWER_KEY_FILE}"
    exit 1
fi

RUN_ROOT="${RUN_ROOT:-../data/smoke_hero_step321_eval}"
FRESH_MODEL_DIR="${FRESH_MODEL_DIR:-false}"
MCP_SERVER_DIR="${MCP_SERVER_DIR:-../mcp_servers/smithery_mcp_servers_0210}"
AGENT_MODE="${AGENT_MODE:-multi}"
AGENT_TOOLS="${AGENT_TOOLS:-virtual}"
USER_MAX_TURNS="${USER_MAX_TURNS:-5}"
MAX_WORKERS="${MAX_WORKERS:-16}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
EVAL_DIMENSIONS="${EVAL_DIMENSIONS:-tool_call,workflow_completion,followup_quality,autonomy,grounding}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-40}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
START_IDX="${START_IDX:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
VIRTUAL_TOOL_MODEL="${VIRTUAL_TOOL_MODEL:-${CANDIDATE_MODEL}}"

slugify_model_name() {
    echo "$1" | tr '/:.' '___' | tr -cd '[:alnum:]_-'
}

latest_file() {
    local pattern="$1"
    shopt -s nullglob
    local matches=( $pattern )
    shopt -u nullglob
    if [ "${#matches[@]}" -eq 0 ]; then
        echo ""
        return 0
    fi
    ls -t "${matches[@]}" | head -n 1
}

model_slug="$(slugify_model_name "${CANDIDATE_MODEL}")"
model_dir="${RUN_ROOT}/${model_slug}"

if [ "${FRESH_MODEL_DIR}" = "true" ] && [ -d "${model_dir}" ]; then
    echo "FRESH_MODEL_DIR=true -> removing existing model dir: ${model_dir}"
    rm -rf "${model_dir}"
fi

mkdir -p "${model_dir}" "${model_dir}/processed_eval"

input_file="${model_dir}/$(basename "${BASE_PREPARED_FILE}")"
answer_key_file="${model_dir}/$(basename "${BASE_ANSWER_KEY_FILE}")"

if [ "${BASE_PREPARED_FILE}" != "${input_file}" ]; then
    cp -f "${BASE_PREPARED_FILE}" "${input_file}"
fi
if [ "${BASE_ANSWER_KEY_FILE}" != "${answer_key_file}" ]; then
    cp -f "${BASE_ANSWER_KEY_FILE}" "${answer_key_file}"
fi

input_base="${input_file%.*}"
if [[ "${input_base}" == *_4prepared ]]; then
    input_base="${input_base%_4prepared}"
elif [[ "${input_base}" == *_prepared ]]; then
    input_base="${input_base%_prepared}"
fi

model_abbr="$(
    MODEL_PATH_ENV="${CANDIDATE_MODEL}" "${PYTHON_BIN}" - <<'PY'
import os
from utils import get_model_abbreviation
print(get_model_abbreviation(os.environ["MODEL_PATH_ENV"]))
PY
)"

echo "=================================================="
echo "Benchmark Pipeline (Step 5 Agent + Eval)"
echo "Candidate model:    ${CANDIDATE_MODEL}"
echo "Judge model:        ${JUDGE_MODEL}"
echo "Virtual tool model: ${VIRTUAL_TOOL_MODEL}"
echo "Agent mode:         ${AGENT_MODE}"
echo "Agent tools:        ${AGENT_TOOLS}"
echo "Model folder:       ${model_dir}"
echo "Input file:         ${input_file}"
echo "Answer key file:    ${answer_key_file}"
echo "OpenRouter URL:     ${OPENROUTER_URL}"
echo "=================================================="

# --- Step 5: Agent Trajectory Generation ---
step5_cmd=(
    "${PYTHON_BIN}" step5_agent.py
    --input_file "${input_file}"
    --model_path "${CANDIDATE_MODEL}"
    --virtual_tool_model "${VIRTUAL_TOOL_MODEL}"
    --base_url "${OPENROUTER_URL}"
    --api_key "${OPENROUTER_API_KEY}"
    --mode "${AGENT_MODE}"
    --tools "${AGENT_TOOLS}"
    --step 3.21m
    --agent openai_agent
    --user_max_turns "${USER_MAX_TURNS}"
    --mcp_server_dir "${MCP_SERVER_DIR}"
    --max_workers "${MAX_WORKERS}"
    --timeout "${TIMEOUT_SECONDS}"
)
if [ -n "${START_IDX}" ]; then
    step5_cmd+=(--start_idx "${START_IDX}")
fi
if [ -n "${BATCH_SIZE}" ]; then
    step5_cmd+=(--batch_size "${BATCH_SIZE}")
fi

"${step5_cmd[@]}"

results_file="$(latest_file "${input_base}_${model_abbr}_multiagent_pfc_results*.jsonl")"
if [ -z "${results_file}" ] || [ ! -f "${results_file}" ]; then
    # Try single-turn naming pattern
    results_file="$(latest_file "${input_base}_${model_abbr}_*_pfc_results*.jsonl")"
fi
if [ -z "${results_file}" ] || [ ! -f "${results_file}" ]; then
    echo "Error: Could not locate Step 5 result file for ${CANDIDATE_MODEL}"
    exit 1
fi

echo "Step 5 output: ${results_file}"

# --- Step 6: Prepare All Eval Prompts (single call) ---
"${PYTHON_BIN}" step6_prepare_eval.py \
    --input_file "${results_file}" \
    --answer_key_file "${answer_key_file}" \
    --dimensions "${EVAL_DIMENSIONS}"

# --- Step 7: Run LLM Judge Evaluation ---
OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" bash step7_eval_completion.sh \
    --input_file "${results_file}" \
    --dimensions "${EVAL_DIMENSIONS}" \
    --multi_turn false \
    --model_path "${JUDGE_MODEL}" \
    --engine openrouter_api \
    --max_tokens "${EVAL_MAX_TOKENS}" \
    --concurrency "${EVAL_CONCURRENCY}"

# --- Step 8: Process and Aggregate Eval Scores ---
"${PYTHON_BIN}" step8_process_eval_scores.py \
    --input_file "${results_file}" \
    --dimensions "${EVAL_DIMENSIONS}" \
    --multi_turn false \
    --output_folder "${model_dir}/processed_eval"

tool_call_scores="$(latest_file "${model_dir}/processed_eval/*_eval_tool_call_*_processed.jsonl")"
workflow_scores="$(latest_file "${model_dir}/processed_eval/*_eval_workflow_completion_*_processed.jsonl")"
grounding_scores="$(latest_file "${model_dir}/processed_eval/*_eval_grounding_*_processed.jsonl")"
followup_scores="$(latest_file "${model_dir}/processed_eval/*_eval_followup_quality_*_processed.jsonl")"
autonomy_scores="$(latest_file "${model_dir}/processed_eval/*_eval_autonomy_*_processed.jsonl")"

"${PYTHON_BIN}" step8_aggregate_eval_scores.py \
    --tool_call_scores "${tool_call_scores}" \
    --workflow_completion_scores "${workflow_scores}" \
    --grounding_scores "${grounding_scores}" \
    --followup_quality_scores "${followup_scores}" \
    --autonomy_scores "${autonomy_scores}" \
    --multi_turn false \
    --output_file "${model_dir}/aggregated_scores_all_dims.jsonl"

"${PYTHON_BIN}" step8_aggregate_eval_scores.py \
    --followup_quality_scores "${followup_scores}" \
    --autonomy_scores "${autonomy_scores}" \
    --multi_turn true \
    --output_file "${model_dir}/aggregated_scores_multi_turn_core.jsonl"

echo ""
echo "Run complete for: ${CANDIDATE_MODEL}"
echo "  Step 5 results:              ${results_file}"
echo "  Processed eval scores folder: ${model_dir}/processed_eval"
echo "  Aggregated (all dims):       ${model_dir}/aggregated_scores_all_dims.jsonl"
echo "  Aggregated (MT core):        ${model_dir}/aggregated_scores_multi_turn_core.jsonl"
