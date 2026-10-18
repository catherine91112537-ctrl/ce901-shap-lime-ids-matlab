# SHAP vs LIME in Network Intrusion Detection (NSL-KDD, MATLAB)

MSc dissertation code: comparative study of explainable AI techniques (SHAP vs LIME) for network intrusion detection on NSL-KDD.

## Structure
- `main.m` – main pipeline (load, preprocess, train, explain)
- `main_per_attack.m` – per-attack-family analysis
- `nslKddShared.m` – shared preprocessing, modelling and attribution helpers
- `make_figures.m` – figure generation
- `KDD Dataset/` – NSL-KDD data (txt/arff)
- `docs/dissertation.docx` – dissertation

## Run
Open MATLAB in this folder and run `main`, then `main_per_attack`, then `make_figures`. Outputs go to `results/` (git-ignored). Set `NSL_KDD_DIR` to use a dataset stored elsewhere.
