## Purpose

These notes tell an AI coding agent how this small NLP workspace is organized, how data flows through the main script, and which conventions are important so changes remain correct on Windows (the development environment used here).

## Key files to inspect

- `vector_space_model.py` — main data-prep script. Loads a Hugging Face dataset, performs stratified sampling by `domain`, and writes a single-line-per-document text file to `OUTPUT_FILE`.
- `logistic_vi_lo_modelv8.joblib` and `tfidf_vectorizer_vi_lov8.joblib` — trained artifacts (joblib format). Treat these as binary model/vectorizer artifacts used by downstream notebooks or services.
- `Untitled.ipynb`, `v0.ipynb` — experiment notebooks. Use them to understand how models/artifacts are consumed.

## Big picture / data flow (concrete)

1. `vector_space_model.py` sets `HF_DATASETS_CACHE` to `D:\HuggingFace` and expects a large dataset (originally ~34GB). It checks free space on `D:` via `psutil.disk_usage("D:\\")` and requires ~40 GB free.
2. It loads dataset `VTSNLP/vietnamese_curated_dataset` with `datasets.load_dataset(..., split='train', columns=['text','domain'])`.
3. Converts to pandas, groups by `domain` and samples each group with `SAMPLE_FRACTION = 0.144` to reduce size (~5GB output).
4. Writes sampled `text` rows (one per line) into `OUTPUT_FILE` (absolute path `D:\vietnamese_5gb_stratified_corpus.txt`).

When editing or extending the pipeline, preserve the above steps or refactor into parameterized functions. Do not silently change absolute paths — they are intentional for this machine's storage layout.

## Environment and dependencies (discoverable)

- OS: Windows (PowerShell is the default shell). Paths and commands should use Windows style.
- Python imports in the repo indicate at minimum: `datasets`, `pandas`, `psutil`. Not present but implied when working with models: `joblib`/`scikit-learn` may be used by notebooks.

Quick install example (agent can suggest to a user):

```powershell
python -m pip install --upgrade pip
python -m pip install datasets pandas psutil
```

## Important project-specific conventions

- Absolute D: drive paths are used throughout (`CACHE_DIR`, `OUTPUT_FILE`). Check for the presence of `D:` and free space before changing them.
- Dataset sampling is intentionally stratified by the `domain` column (`groupby(DOMAIN_COLUMN).apply(lambda x: x.sample(...))`). Any change to sampling must preserve stratification unless explicitly required.
- Console messages are written in Vietnamese in `vector_space_model.py`; maintain or document language choices when adding user-facing output.

## Reproducible run / debugging steps (concrete)

1. Ensure `D:` has enough free space (>= 40 GB). The script enforces this — tests will fail locally otherwise.
2. From PowerShell run the data-prep script:

```powershell
python "d:\NLP\machine learning\vector_space_model.py"
```

3. If you need to iterate faster, reduce `SAMPLE_FRACTION` temporarily and run again.

4. To inspect model artifacts, open notebooks `v0.ipynb` or `Untitled.ipynb` rather than editing binary `.joblib` files directly.

## Patterns and anti-patterns observed

- Pattern: single-responsibility script that combines configuration, environment checks, dataset download, sampling, and writing to disk — useful for quick runs but fragile for reuse. If refactoring, split config, IO, and sampling logic into functions.
- Anti-pattern to avoid: hardcoding new absolute paths without verifying `D:` state. Many defaults assume `D:` exists.

## Examples to reference in edits

- Disk check and requirement:
  - `disk_usage = psutil.disk_usage("D:\\")` and `if disk_usage.free < 40: raise OSError(...)`
- Dataset load and sampling:
  - `dataset = load_dataset(DATASET_ID, split='train', columns=[TEXT_COLUMN, DOMAIN_COLUMN])`
  - `sampled_df = df.groupby(DOMAIN_COLUMN, group_keys=False).apply(lambda x: x.sample(frac=SAMPLE_FRACTION, random_state=42))`

## When modifying or adding tests

- Fast unit tests should mock `datasets.load_dataset` and `psutil.disk_usage` so tests run without large downloads or disk requirements. Use small DataFrame fixtures to validate grouping and sampling logic.

## Safety & verification checks an agent should run before proposing changes

- Verify `D:` exists and is writable when changing `CACHE_DIR` or `OUTPUT_FILE`.
- Preserve `random_state=42` for deterministic sampling unless the change intentionally removes determinism.
- Do not remove `columns=[TEXT_COLUMN, DOMAIN_COLUMN]` from `load_dataset` unless the downstream code is updated accordingly.

---

If anything is unclear (for example: which notebook reproduces training, or whether models are loaded by another service), tell me what to inspect next and I will update these instructions. Iteration welcome.
