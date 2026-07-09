#!/bin/bash
# Sequential RAG + Ensemble Runner
set -euo pipefail

# --- CONFIGURATION ---
: "${NVAPI_KEY:?Set NVAPI_KEY before running this script (export NVAPI_KEY=...)}"
APIKEY="$NVAPI_KEY"
TEAM_ID="CanSA"
DATA_DIR="./raw.txt+gold.json/"
RAG_INDEX="./rag_index"
TEST_DIR="./Test_Summaries/"
VAL_DIR="./Val_Summaries/"
SCRIPT="./medexact_rag_v9_gpuG.py"
BASE_URL="https://integrate.api.nvidia.com/v1"

# --- FAIL-FAST ASSERTIONS ---
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script $SCRIPT not found!"
    exit 1
fi

if [[ ! -d "$RAG_INDEX" ]]; then
    echo "ERROR: FAISS index directory $RAG_INDEX not found! Run build-index first."
    exit 1
fi

MODELS=(
    "gemma3-27b|google/gemma-3-27b-it"
    "kimi-k2|moonshotai/kimi-k2-instruct"
    "llama33|meta/llama-3.3-70b-instruct"
    "mistral-large3|mistralai/mistral-large-3-675b-instruct-2512"
    "qwen35|qwen/qwen3.5-122b-a10b"
    "phi4-multi|microsoft/phi-4-multimodal-instruct"
)

echo "STEP 3: Sequential Ensemble Merge (RAG + Zero-shot) on Val..."
for entry in "${MODELS[@]}"; do
    IFS="|" read -r NAME MODEL <<< "$entry"
    ENS_DIR="output_${NAME}_ENS_val"
    BASELINE="output_${NAME}_Zeroshot_val"

    # Defensive check: Did the zero-shot phase actually produce the baseline?
    if [[ ! -d "$BASELINE" ]]; then
        echo "WARNING: Baseline directory $BASELINE not found. Merging will fail. Skipping $NAME Val..."
        continue
    fi

    echo "------------------------------------------"
    echo "[ENS VAL] $NAME -> $ENS_DIR"

    # Isolate failure so one bad model doesn't kill the script
    set +e
    python "$SCRIPT" run \
        --provider openai --base-url "$BASE_URL" --api-key "$APIKEY" --model "$MODEL" --data-dir "$VAL_DIR" \
        --output-dir "$ENS_DIR" --index-path "$RAG_INDEX" \
        --top-k 5 \
        --team-id "$TEAM_ID" \
        --merge-baseline "$BASELINE" --skip-done --timeout 1000 --api-delay 15

    if [ $? -ne 0 ]; then
        echo "WARNING: $NAME Val RAG failed. Moving to next model..."
    fi
    set -e
done

echo "------------------------------------------"
echo "All RAG steps completed for all models."
