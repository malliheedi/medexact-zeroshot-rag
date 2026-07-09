#!/bin/bash
# Sequential Zero-Shot Runner
set -euo pipefail

# --- CONFIGURATION ---
: "${NVAPI_KEY:?Set NVAPI_KEY before running this script (export NVAPI_KEY=...)}"
APIKEY="$NVAPI_KEY"
TEAM_ID="CanSA"
DATA_DIR="./raw.txt+gold.json/"
TEST_DIR="./Test_Summaries/"
SCRIPT="./medexact_batch_zeroshot_v14G.py"
BASE_URL="https://integrate.api.nvidia.com/v1"

MODELS=(
    "gemma3-27b|google/gemma-3-27b-it"
    "kimi-k2|moonshotai/kimi-k2-instruct"
    "llama33|meta/llama-3.3-70b-instruct"
    "mistral-large3|mistralai/mistral-large-3-675b-instruct-2512"
    "qwen35|qwen/qwen3.5-122b-a10b"
    "phi4-multi|microsoft/phi-4-multimodal-instruct"
)

echo "Starting sequential Val Zero-Shot pipeline..."

for entry in "${MODELS[@]}"; do
    IFS="|" read -r NAME MODEL <<< "$entry"
    OUTDIR="output_${NAME}_Zeroshot_val"

    echo "------------------------------------------"
    echo "[RUNNING VAL] $NAME -> $OUTDIR"

    # We remove set -e temporarily around the python call so a single model crash
    # does not kill the entire overnight script.
    set +e
    python "$SCRIPT" \
        --provider openai \
        --base-url "$BASE_URL" \
        --api-key "$APIKEY" \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTDIR" \
        --ids val.txt \
        --team-id "$TEAM_ID" \
        --skip-done \
        --timeout 1000 \
        --api-delay 15

    if [ $? -ne 0 ]; then
        echo "WARNING: $NAME Val failed. Moving to next model..."
    fi
    set -e
done

echo "------------------------------------------"
echo "All Zero-Shot runs completed."
