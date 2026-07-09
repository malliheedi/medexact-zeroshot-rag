#!/bin/bash
# Master Runner: Zero-Shot -> then -> RAG Pipeline
# Removed set -e so the pipeline continues even if a phase throws an error

echo "--- [$(date)] STARTING ZERO-SHOT PHASE ---"
./run_all_apiG.sh 2>&1 | tee -a pipelineZero.log || echo "WARNING: Zero-Shot phase reported errors."

echo "--- [$(date)] ZERO-SHOT COMPLETE. STARTING RAG PHASE ---"
./run_pipeline_fullG.sh 2>&1 | tee -a pipelineRAG.log || echo "WARNING: RAG phase reported errors."

echo "--- [$(date)] ALL PHASES FINISHED ---"
