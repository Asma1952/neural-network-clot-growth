# neural-network-clot-growth
Code and data for "A digital twin framework integrating in vivo clot growth with in vitro thrombin generation for patient-specific management of NOAC therapy"
---

## Repository contents

| File | Description |
|---|---|
| `train_and_tune.R` | Benchmark model training, two-stage SPNN hyperparameter tuning, and final 4000-epoch model |
| `analysis_and_visualization.R` | Repeated CV, partial nested CV, ablation studies, SHAP, ICE, PDPs, residuals, and gradient visualization |
| `sensitivity_analysis.R` | Lambda sensitivity, input perturbation stability, gradient analysis, and monotonicity verification |
| `NEW_data_clean.rds` | Simulated patient dataset (N=1086) |
| `spnn_final_model.pt` | Trained tuned SPNN weights |
| `spnn_default_model.pt` | Trained default SPNN weights |



## Requirements

R version 4.x with the following packages:

```r
install.packages(c("tidyverse", "caret", "randomForest",
                   "xgboost", "e1071", "glmnet", "kernlab"))

# torch
install.packages("torch")

# catboost (not on CRAN)
devtools::install_github("catboost/catboost",
                         subdir="catboost/R-package")

# fastshap (not on CRAN)
devtools::install_github("bgreenwell/fastshap")
```

---

## How to reproduce

Run the scripts in order:

```r
source("train_and_tune.R")           # ~4-5 hours
source("analysis_and_visualization.R") # ~18-22 hours
source("sensitivity_analysis.R")     # ~30 minutes
```

All outputs (CSV files and PNG plots) are saved to the working directory.

---

## Reproducibility

| Setting | Value |
|---|---|
| R seed | `set.seed(8547)` |
| torch seed | `torch_manual_seed(8547)` |
| Split | 80/20 stratified |
| Train set | n = 869 |
| Test set | n = 217 |

---

## Notes

- `analysis_and_visualization.R` and `sensitivity_analysis.R` require the RDS and PT files saved by `train_and_tune.R`
- `sensitivity_analysis.R` is fully standalone and reloads all objects from disk
- Torch model weights require the `torch` R package to load
