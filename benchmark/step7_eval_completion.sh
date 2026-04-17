#!/bin/bash
#
# Step 7: Run LLM judge evaluation for one or more eval dimensions.
#
# Discovers prepared prompt files by convention:
#   {base}_eval_{dimension}_prepared.jsonl
# where {base} is --input_file with .jsonl stripped.
#
# Usage:
#   bash step7_eval_completion.sh \
#     --input_file path/to/results.jsonl \
#     --dimensions "all" \
#     --multi_turn "true" \
#     --model_path "openai/gpt-oss-120b" \
#     --engine "openrouter_api" \
#     --max_tokens 4096 \
#     --concurrency 40
#

# --- Defaults ---
input_file=""
dimensions="all"
model_path="openai/gpt-oss-120b"
engine="openrouter_api"
max_tokens=4096
multi_turn="false"
concurrency=50

# --- Color definitions ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)  input_file="$2";  shift 2 ;;
        --dimensions)  dimensions="$2";  shift 2 ;;
        --model_path)  model_path="$2";  shift 2 ;;
        --engine)      engine="$2";      shift 2 ;;
        --max_tokens)  max_tokens="$2";  shift 2 ;;
        --multi_turn)  multi_turn="$2";  shift 2 ;;
        --concurrency) concurrency="$2"; shift 2 ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            exit 1
            ;;
    esac
done

if [ -z "$input_file" ]; then
    echo -e "${RED}Error: --input_file is required.${NC}"
    echo "Usage: bash step7_eval_completion.sh --input_file <path.jsonl> [--dimensions all] [--model_path ...] [--engine openrouter_api] [--multi_turn false]"
    exit 1
fi

# Derive base path by stripping .jsonl extension
base_path="${input_file%.jsonl}"

# --- Expand dimensions ---
if [ "$dimensions" == "all" ]; then
    if [ "${multi_turn}" == "true" ]; then
        dimensions="followup_quality,autonomy,grounding"
    else
        dimensions="tool_call,workflow_completion,grounding,followup_quality"
    fi
fi

IFS=',' read -ra DIM_ARRAY <<< "$dimensions"

echo -e "${BLUE}[Step 7] Evaluation Completion${NC}"
echo -e "  Input file:  ${input_file}"
echo -e "  Dimensions:  ${dimensions}"
echo -e "  Multi-turn:  ${multi_turn}"
echo -e "  Model:       ${model_path}"
echo -e "  Engine:      ${engine}"
echo -e "  Concurrency: ${concurrency}"
echo ""

if [ "$engine" != "openai" ] && [ "$engine" != "openrouter_api" ]; then
    echo -e "${RED}Error: Only 'openai' and 'openrouter_api' engines are supported.${NC}"
    exit 1
fi

# --- Run each dimension ---
FAILED=0
for dim in "${DIM_ARRAY[@]}"; do
    dim=$(echo "$dim" | xargs)  # trim whitespace
    prepared_input_file="${base_path}_eval_${dim}_prepared.jsonl"

    if [ ! -f "$prepared_input_file" ]; then
        echo -e "${YELLOW}[Step 7] Skipping '${dim}': prepared file not found at ${prepared_input_file}${NC}"
        continue
    fi

    echo -e "${BLUE}[Step 7] Running evaluation for dimension: ${dim}${NC}"
    echo -e "  Input: ${prepared_input_file}"

    python step7_eval_endpoint.py \
        --input_file "${prepared_input_file}" \
        --model_path "${model_path}" \
        --engine "${engine}" \
        --step "eval_${dim}" \
        --max_tokens ${max_tokens} \
        --concurrency ${concurrency}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[Step 7] Completed: ${dim}${NC}"
    else
        echo -e "${RED}[Step 7] Failed: ${dim}${NC}"
        FAILED=$((FAILED+1))
    fi
    echo ""
done

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}[Step 7] ${FAILED} dimension(s) failed.${NC}"
    exit 1
else
    echo -e "${GREEN}[Step 7] All dimensions completed successfully.${NC}"
fi
