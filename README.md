# MedExACT Zero-Shot and RAG Pipeline

This repository contains the training-free extraction pipeline used to produce
the results reported in "Infrastructure-Aware Clinical Decision Extraction:
A Zero-Shot and Retrieval-Augmented Pipeline for Span-Level DICTUM
Classification Across Heterogeneous Hardware." The pipeline extracts and
classifies clinical decisions from ICU discharge summaries under the
nine-category DICTUM taxonomy, using the MedDec dataset built on MIMIC-III.

## Repository Structure

The pipeline evolved across two hardware phases, and the file names reflect
this history. Since a reader may not know which script produced which table
in the paper, the lineage is described explicitly below rather than left
implicit in the file names.

```
.
├── master_run.sh                     # Entry point: runs zero-shot, then RAG, sequentially
├── medexact_batch_zeroshot_v14G.py   # Zero-shot pipeline, NVIDIA NIM API (Phase 2)
├── medexact_rag_v9_gpuG.py           # RAG pipeline, NVIDIA NIM API (Phase 2)
├── run_all_apiG.sh                   # Sequential zero-shot runner across models
├── run_pipeline_fullG.sh             # Sequential RAG + ensemble merge runner
├── legacy/
│   ├── medexact_batch_zeroshot_v14.py  # Original Ollama/dual-provider zero-shot script (Phase 1)
│   ├── medexact_rag_v9_gpu.py          # Original Ollama-only RAG pipeline (Phase 1)
│   └── medexact_rag_v9_gpuVT.py        # Extended Ollama-only variant, local models only
└── README.md
```

The two scripts in the repository root, `medexact_batch_zeroshot_v14G.py` and
`medexact_rag_v9_gpuG.py`, are the scripts actually invoked by
`run_all_apiG.sh` and `run_pipeline_fullG.sh`, and are therefore the ones
responsible for the Phase 2 NIM API results in Tables 2, 4, and 5 of the
paper. The three files under `legacy/` correspond to the original A6000
Ollama pipeline used for the Phase 1 results in Table 1
(`medexact_batch_zeroshot_v14.py` and `medexact_rag_v9_gpu.py`) and a
separate Ollama-only extension that was not used to generate any table in
the paper (`medexact_rag_v9_gpuVT.py`). They are kept for completeness and
transparency, but should not be treated as reproducing the Phase 2 numbers.

## Requirements

The pipeline requires Python 3.10 or later. Dependencies install
automatically on first run via `pip install --break-system-packages`, and
include `rich`, `sentence-transformers`, `faiss-gpu-cu12` (or `faiss-cpu`
where no GPU is available), and `numpy<2`. A CUDA-capable GPU is recommended
for building the FAISS index, though the RAG pipeline falls back to CPU
inference if a GPU is not found.

## API Key Configuration

Both shell scripts read the NVIDIA NIM API key from an environment variable
rather than embedding it in the script. Before running either script, export
the key:

```bash
export NVAPI_KEY="your-nvidia-nim-api-key-here"
```

Both scripts exit with an error if `NVAPI_KEY` is unset. This is a required
step, since a key committed to version history remains recoverable even
after a later commit removes it.

## Data Layout

The pipeline expects the following directory layout, matching the MedDec
release:

```
raw.txt+gold.json/    # discharge_id.txt (raw note) + discharge_id.json (gold annotations)
Val_Summaries/         # 53-document validation subset
Test_Summaries/        # held-out test subset
rag_index/              # built by the build-index command below
```

The MedDec dataset itself is available at
https://github.com/CLU-UML/MedDec and is not redistributed here, since
access to the underlying MIMIC-III text requires credentialed PhysioNet
access.

## Usage

### Full pipeline (recommended)

Once the FAISS index is built (see below) and `NVAPI_KEY` is exported, the
entire pipeline runs end to end with:

```bash
export NVAPI_KEY="your-key-here"
bash master_run.sh
```

This runs the zero-shot phase to completion first, then the RAG and
ensemble-merge phase, logging each phase separately to `pipelineZero.log`
and `pipelineRAG.log`. Since `master_run.sh` does not stop on a phase
error, a failure in one model's zero-shot run does not block the RAG phase
for the remaining models, though the eventual ensemble merge for that
specific model is skipped due to the missing-baseline check already built
into `run_pipeline_fullG.sh`.

### Running phases individually

The steps below remain available for rerunning a single phase, for example
after adding a new model to the array.

**1. Build the FAISS retrieval index.** The RAG pipeline retrieves
in-context exemplars from the 350-document training subset. The index must
be built once before any RAG run:

```bash
python medexact_rag_v9_gpuG.py build-index \
    --train-dir ./raw.txt+gold.json/ \
    --index-path ./rag_index \
    --chunk-size 2000 \
    --overlap 200
```

**2. Run zero-shot extraction:**

```bash
export NVAPI_KEY="your-key-here"
bash run_all_apiG.sh
```

This sequentially runs each model in the `MODELS` array against the
53-document validation set, writing output to
`output_<model>_Zeroshot_val/`. Each call enforces a 15-second delay
between API requests, consistent with the stability-first configuration
described in Section 3.5 of the paper.

**3. Run RAG-augmented ensemble extraction:**

```bash
export NVAPI_KEY="your-key-here"
bash run_pipeline_fullG.sh
```

This step requires the zero-shot output directories from step 2 to exist,
since the ensemble merge step combines RAG predictions with the zero-shot
baseline for each model. Merged predictions are written to
`output_<model>_ENS_val/`.

**4. Evaluate.** Both `medexact_batch_zeroshot_v14G.py` and
`medexact_rag_v9_gpuG.py` compute Span F1, Token F1, and Base F1
automatically at the end of each run when gold annotations are present
alongside the input text files, and print a summary table to the terminal.

## Prompt Template

The system prompt used for span extraction is defined as `RAG_SYSTEM_PROMPT`
in `medexact_rag_v9_gpuG.py` (and correspondingly `SYSTEM_PROMPT` in the
zero-shot script) and is also reproduced in Appendix A of the paper. Both
enforce a strict JSON output schema over the nine DICTUM categories and
instruct the model to preserve de-identification tokens verbatim. Since
some models (Qwen 3.5, DeepSeek) emit internal reasoning chains that
corrupt this schema, the pipeline prepends a `/no_think` directive and sets
`enable_thinking: false` in the API payload for those model families.

## Known Limitations

The pipeline's zero-span failure mode under HTTP 504 gateway timeouts is
described in Sections 3.5 and 5.1 of the paper as silent poisoning.
Chunk-level logging of these failures was not implemented for the
validation runs reported in this repository's results, so exact failure
counts are not available for post hoc audit. Per-DICTUM category
performance breakdowns are also not currently produced by the evaluation
code and are left for future work.

## Citation

If this pipeline is used in further work, please cite the paper this
repository accompanies. Citation details will be added upon publication.

## License

MIT License. See `LICENSE` for details.
