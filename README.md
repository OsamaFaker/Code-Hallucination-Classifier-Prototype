# Code-Hallucination-Classifier Prototype

> [!WARNING]
> **Running this code executes untrusted, model-generated Python.**
> The oracle executor (`src/oracle_executor.py`) and the dataset generator
> (`scripts/generate_oracle_dataset.py`) run model-generated code in a plain child
> process (`subprocess.run([sys.executable, file])`) with **your** filesystem and
> network privileges and a 5-second timeout. This is **not a sandbox**. Run the
> classifier and the generator only inside a disposable container or VM.
> Reproducing the paper's numbers (`analysis/paper_analysis.py`, `scripts/evaluate.py`)
> executes no generated code and is safe to run anywhere.

Artifact for the ACM TOSEM submission *"Distinguishing Errors from Hallucinations in
LLM-Generated Code: A Grounding-Based Theoretical Framework"* (see [Citation](#citation)).

A static-plus-LLM pipeline that classifies LLM-generated Python code.
Each code snippet is assigned one of three labels:

| Label | Meaning |
|---|---|
| `VALID` | Code is correct and all tests pass |
| `HALLUCINATION` | Code references a fabricated module, function, or identifier that does not exist in the Python Ground Truth |
| `GROUNDED_ERROR` | Code uses real Python constructs but applies them incorrectly (wrong types, wrong arguments, failed assertions) |

---

## Architecture

```
Input code
    │
    ▼
Phase 1 — Static checkers (fabrication · attribute · arity · signature · misuse · logic)
    │  rule-based rejection without LLM calls
    ▼
Phase 2 — LLM semantic verifier (qwen2.5-coder:7b  or  llama3:latest)
    │  for samples that pass the static funnel
    ▼
Phase 3 — Oracle executor (plain subprocess, see warning above)
    │  runs test assertions; overrides any VALID verdict that fails at runtime
    ▼
Predicted label  +  confidence  +  checker_source  +  justification
```

The fabrication checker is an **allowlist test**: a module is flagged when it is not in
the Python Ground Truth registry and cannot be imported locally. It does **not** query PyPI.

---

## Validity and limitations

Read these before interpreting the results.

- **Labels are execution-derived** (from running the code against the MBPP tests) and
  have **not been validated by human annotation**.
- **The full-pipeline result (accuracy 0.938) is an upper bound.** Phase 3 uses the same
  execution routine that produced the labels. Without Phase 3, accuracy is
  **0.662 (Qwen)** and **0.651 (Llama 3)**.
- **A single rule is a strong baseline.** "Predict HALLUCINATION if the code imports any
  non-standard-library module, otherwise VALID" reaches accuracy **0.664** and
  HALLUCINATION F1 **0.990**.
- **Both verifiers give identical predictions** because the verifier rarely changes a
  verdict: it returns a non-VALID verdict for **5 samples under Qwen and 0 under Llama 3**.
  The agreement says nothing about verifier quality and does not show that the pipeline is independent of the model.
- **Only 36 of the 150 HALLUCINATION samples are verified fabrications.** The dataset
  generator asks the model to import one of 15 invented library names. 11 of these names
  are registered on PyPI (checked 11 Oct 2026, `python analysis/check_pypi.py`); only
  `calctools`, `codeutils`, `listops` and `sortinglib` are not. The other samples count
  as hallucinations only relative to the Ground Truth registry.
- **Type-confusion attributes are labelled HALLUCINATION.** For example, `list.push`,
  `list.add` and `str.append` are reported as hallucinations. Under Condition 2 of the paper they
  are grounded errors; the labelling rule has the same limitation (Section 5.1).
- **Generation is not bit-reproducible.** The dataset was generated with Ollama at
  temperature 0.0 (hallucinated), 0.2 (direct) and 0.3 (grounded error) **without a
  seed**; only class balancing is seeded (`seed=42`). The committed dataset
  `results/mbpp_oracle_450_balanced.json` is the reference.

---

## Dataset

The evaluation uses an **execution-oracle dataset** built from [MBPP](https://github.com/google-research-datasets/mbpp) problems.
Each sample's label is determined by executing the code against the MBPP test assertions (see [Validity and limitations](#validity-and-limitations)).

| Property | Value |
|---|---|
| Source | MBPP (974 problems available) |
| Problems used | 164 (early-stop once all classes hit 150 samples) |
| Total samples | **450** |
| Label balance | **150 VALID · 150 HALLUCINATION · 150 GROUNDED_ERROR (33% each)** |
| Generator model | `qwen2.5-coder:7b` |
| File | `results/mbpp_oracle_450_balanced.json` |

### Generating the dataset yourself

⚠️ Executes model-generated code — use a disposable container or VM. The output will differ
from the committed dataset (unseeded sampling, see above).

```bash
python scripts/generate_oracle_dataset.py \
  --mbpp-path /path/to/mbpp.jsonl \
  --problems 974 \
  --retries 3 \
  --max-each 150 \
  --balance \
  --target-each 150 \
  --output results/mbpp_oracle_450_balanced.json
```

| Flag | Description |
|---|---|
| `--mbpp-path` | Local MBPP `.jsonl` file (974 problems) |
| `--problems N` | Maximum problems to draw from |
| `--retries K` | Attempts per strategy until oracle label matches the intended label |
| `--max-each N` | Early-stop once every class has N confirmed samples |
| `--balance` | Undersample majority classes for equal label distribution |
| `--target-each N` | Samples per label after balancing |

---

## Requirements

- Python ≥ 3.10 for `analysis/paper_analysis.py` (uses `sys.stdlib_module_names`);
  the classifier itself supports Python ≥ 3.9.
- `requirements.txt` pins the environment in which the tests and the analysis were
  verified for this archive (Python 3.14.0, Windows 11).
- [Ollama](https://ollama.com) running locally on port 11434 (only for classification
  and dataset generation).

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q             # 66 tests
```

### Model versions

| Tag | Role | Ollama ID | Manifest digest (sha256) |
|---|---|---|---|
| `qwen2.5-coder:7b` | dataset generator; verifier (Qwen run) | `dae161e27b0e` | `dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364` |
| `llama3:latest` | verifier (Llama 3 run) | `365c0bd3c000` | `365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1` |

Digests from `ollama list` on the machine that produced the committed results (models
pulled 15–16 May 2026). Tags are mutable; pull by digest to get the same weights.

```bash
ollama pull qwen2.5-coder:7b
ollama pull llama3:latest
```

---

## Reproducing the paper

All commands below read only the committed files in `results/`. They use no model and
execute no generated code.

| Paper item | Command | Output to read |
|---|---|---|
| Table 4 — full-pipeline metrics with 95% CIs | `python analysis/paper_analysis.py` | block "Table 4" |
| Table 5 — ablation and reference points | `python analysis/paper_analysis.py` | block "Table 5" |
| Table 6 — error analysis (GROUNDED_ERROR → HALLUCINATION) | `python analysis/paper_analysis.py` | block "Table 6" |
| Section 5.3 — phase decomposition and 36/36 verified recall | `python analysis/paper_analysis.py` | block "Section 5.3" |
| Section 5.3 — PyPI status of the 15 fabrication-list names | `python analysis/check_pypi.py` | (queries pypi.org) |
| Full-pipeline accuracy / confusion matrix per model | `python scripts/evaluate.py results/mbpp_oracle_450_balanced_qwen_enhanced_results.json results/mbpp_oracle_450_balanced_llama3_enhanced_results.json` | |

`python analysis/paper_analysis.py --json` prints every number in machine-readable form.
Confidence intervals are 95% percentile intervals from a cluster bootstrap over
`prompt_id` (164 problems, 10,000 replicates, `numpy.random.default_rng(20261011)`).
A proportion whose point estimate is 1 instead gets an exact Clopper–Pearson interval.

---

## Running the classifier

⚠️ Phase 3 executes model-generated code — use a disposable container or VM.

```bash
# Qwen 2.5 Coder
python scripts/classify.py \
  --input results/mbpp_oracle_450_balanced.json \
  --model qwen \
  --use-cov

# Llama 3
python scripts/classify.py \
  --input results/mbpp_oracle_450_balanced.json \
  --model llama3 \
  --use-cov
```

The committed reference outputs are:

```
results/mbpp_oracle_450_balanced_qwen_enhanced_results.json
results/mbpp_oracle_450_balanced_llama3_enhanced_results.json
```

Without `--output`, a new run writes `results/<input-stem>_<model>_results.json`
(for example `results/mbpp_oracle_450_balanced_qwen_results.json`), so it never
overwrites the reference files.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | built-in default | Input dataset JSON |
| `--model` | `qwen` | Verifier model: `qwen` or `llama3` |
| `--output PATH` | auto | Override output path |
| `--workers N` | `3` | Parallel LLM-verifier workers |
| `--use-cov` | off | Enable Chain-of-Verification 3-way semantic verifier (used for the paper) |

---

## Results

Evaluated on the 450-sample balanced oracle dataset (150 per class). Qwen 2.5 Coder 7B
and Llama 3 8B produce **identical predictions**, so one table covers both. 95% CIs are
from `analysis/paper_analysis.py`.

**Full pipeline (upper bound, see above): accuracy 0.938 [0.909, 0.964] · macro F1 0.938 [0.909, 0.965]**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| VALID | 1.000 [0.974, 1.000] | 0.940 [0.891, 0.980] | 0.969 [0.942, 0.990] |
| GROUNDED_ERROR | 0.963 [0.920, 0.993] | 0.873 [0.800, 0.938] | 0.916 [0.871, 0.955] |
| HALLUCINATION | 0.867 [0.808, 0.927] | 1.000 [0.976, 1.000] | 0.929 [0.894, 0.962] |

Confusion matrix (rows = true label, columns = predicted):

```
                   VALID   GROUNDED_ERROR   HALLUCINATION
TRUE VALID           141                5               4
TRUE GROUNDED_ERROR    0              131              19
TRUE HALLUCINATION     0                0             150
```

### Ablation and reference points

| Configuration | Accuracy [95% CI] | GROUNDED_ERROR F1 | HALLUCINATION F1 |
|---|---|---|---|
| Uniform guessing | 0.333 | — | — |
| Non-stdlib-import rule | 0.664 [0.614, 0.715] | 0.000 | 0.990 |
| Phase 1 only | 0.651 [0.602, 0.700] | 0.025 | 0.929 |
| Phases 1–2 (Qwen) | 0.662 [0.612, 0.711] | 0.086 | 0.929 |
| Phases 1–2 (Llama 3) | 0.651 [0.602, 0.700] | 0.025 | 0.929 |
| Full pipeline (both verifiers) | 0.938 [0.909, 0.964] | 0.916 | 0.929 |

### Observations

- HALLUCINATION recall is 1.000 on this dataset (150/150), and 36/36 on the verified
  fabrications (95% Clopper–Pearson lower bound 0.903). This holds for these labels
  only; see [Validity and limitations](#validity-and-limitations).
- VALID precision is 1.000 (141/141). This holds by construction: Phase 3
  overrides any VALID verdict that fails the tests that define the labels.
- Most of the gain over the baselines comes from Phase 3, which reuses the labelling
  routine.
- The main remaining confusion is GROUNDED_ERROR predicted as HALLUCINATION (19 cases:
  fabrication checker 12, misuse 4, function_name 2, attribute 1; by exception
  AssertionError 12, TypeError 4, NameError 3).

---

## Output schema

Each entry in the classified JSON files contains the original sample fields plus:

| Field | Type | Description |
|---|---|---|
| `predicted_label` | string | `VALID`, `GROUNDED_ERROR`, or `HALLUCINATION` |
| `confidence` | float | Combined pipeline confidence (0–1) |
| `checker_source` | string | Which component produced the verdict (static checker name, `cov_verifier`, or `oracle_exec`) |
| `verifier_justification` | string | One-sentence LLM justification (LLM-verified samples only) |
| `latency_ms` | float | Per-call LLM latency in milliseconds |
| `taxonomy_tag` | string | Fine-grained classification tag (verifier verdict for Phase 2/3 samples) |

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/classify.py` | Run the classifier on a dataset |
| `scripts/evaluate.py` | Compute accuracy, F1, and confusion matrix; optionally compare two result files |
| `scripts/calibrate.py` | Fit Platt-scaling confidence calibrators from existing results |
| `scripts/generate_oracle_dataset.py` | Build a new execution-oracle dataset from MBPP |
| `analysis/paper_analysis.py` | Reproduce the paper's tables from the committed results |
| `analysis/check_pypi.py` | Check the fabrication-list names against PyPI |
| `harness.py` | Evaluation harness (called by `classify.py`) |

---

## Citation

If you use this artifact, please cite the paper (see `CITATION.cff`):

> Osama Faker and Sacip Toker. *Distinguishing Errors from Hallucinations in
> LLM-Generated Code: A Grounding-Based Theoretical Framework.* ACM Transactions on
> Software Engineering and Methodology (submitted).

## License

MIT — see [LICENSE](LICENSE).
