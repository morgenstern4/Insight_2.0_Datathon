[README.md](https://github.com/user-attachments/files/32531356/README.md)
# Insight 2.0 — Vital Status Prediction

**Team ABC · IASDS Datathon 2026 — Online Qualifier**

A censored-survival treatment of what looks, on the surface, like an ordinary binary
classification task: predicting `vital_status` (`Dead` / `Alive`) for lung cancer patients
from a SEER-style registry extract.

---

## TL;DR

`vital_status` isn't a clean label — it records whether a patient had died *by the registry
cut-off*, which means it confounds how sick a patient was with how long they'd been observed.
A patient diagnosed in 2013 has ~10 years of follow-up; one diagnosed in 2023 has ~1. Treating
this as an ordinary classification problem and throwing gradient boosters at it produces a
number, but not an honest one.

This notebook instead:

1. **Derives and tests** the censored-survival structure implied by the data (a
   complementary log-log link with `log(follow-up time)` as a covariate), and uses it to
   justify feature engineering and monotone constraints rather than hoping a booster
   rediscovers it.
2. **Seals off a 20% audit set** before any tuning, model selection, or threshold choice, so
   the final reported score is not the product of its own validation.
3. **Separates tuning from decision-making**: hyperparameters are optimized on log loss (a
   proper scoring rule), calibration is checked separately, and the Dead/Alive threshold is
   chosen last, on fixed probabilities.
4. **Quantifies how much of its own performance is real medicine vs. a registry artifact** —
   an ablation and a within-diagnosis-year AUC decomposition (§7) — rather than reporting a
   single pooled AUC and moving on.
5. **Reports what didn't work** (§8.4) and states its limitations (§9) as plainly as its
   results.

No AutoML is used anywhere; every model, feature, and search space is hand-specified.

---

## Repository contents

| File | Description |
|---|---|
| `team_ABC_notebook.ipynb` | The full pipeline — data understanding, feature engineering, model ladder, tuning, audit evaluation, and final submission generation. Deterministic; running it top to bottom regenerates `submission.csv`. |
| `submission.csv` | Final predictions — 36,000 rows, columns `patient_id`, `vital_status` (`Dead`/`Alive`). |

The notebook expects `train.csv` and `test.csv` (the competition's SEER-derived extract, not
included in this repo — see [Data](#data)) in a `DATA_DIR` set near the top of the notebook.

---

## Methodology

The notebook is organized into nine sections, each building on evidence established in the
one before it:

| § | Section | What it establishes |
|---|---|---|
| 1 | Setup & the size of the prize | Measures what a conventional "tune three boosters against F1" pipeline actually scores, to calibrate expectations for every later gain |
| 2 | Understanding the data-generating process | Confirms the censored-survival structure (cloglog vs. `log t`), checks train/test are the same distribution (adversarial validation), quantifies the missingness mechanism and the irreducible (label-noise) error floor |
| 3 | Validation protocol | Declares the 20% audit-set split and the CV scheme **before** any model is fit |
| 4 | Feature engineering | Survival-time features, sentinel-code splitting (SEER reserves codes like `98`/`99` as flags, not magnitudes), ordinal T/N/M staging, node ratios — each justified by §2, nothing speculative |
| 5 | A ladder of models | Constant → parametric cloglog GLM → regularized logistic regression → monotone-constrained LightGBM → unconstrained LightGBM → CatBoost → XGBoost, each required to beat the previous rung by more than its bootstrap confidence interval |
| 6 | Tuning, calibration, and the decision rule | Learning-curve diagnosis, Optuna search on log loss (not F1), calibration check, and threshold selection — kept as four separate problems |
| 7 | What did the model actually learn? | Ablates the censoring features and measures AUC *within* diagnosis year, to separate genuine clinical discrimination from a registry-length artifact |
| 8 | Final model selection & the honest estimate | Model-family pool fixed *a priori* (one representative per boosting library), blend weights cross-fitted on log loss, evaluated once on the sealed audit set |
| 9 | Limitations | Stated plainly — see [Limitations](#limitations) below |

### Final model

A blend of three gradient-boosting families, chosen as a fixed pool before any score was
seen (LightGBM appears once, tuned and monotone-constrained; CatBoost and XGBoost use
hyperparameters from a completed Optuna study rather than being re-searched here):

- **LightGBM** — monotone-constrained (risk non-decreasing in stage, age, metastatic burden,
  follow-up time; non-increasing in surgical resection), tuned by Optuna on log loss
- **CatBoost** — ordered target statistics for categoricals, decorrelates well with LightGBM
- **XGBoost** — level-wise tree growth, added purely for ensemble diversity

Blend weights are cross-fitted on log loss rather than hand-set (final fit: LightGBM 0.22 /
CatBoost 0.66 / XGBoost 0.12).

---

## Results

Measured on the **sealed 20% audit set** — 4,800 rows that influenced no modeling, tuning,
feature, or threshold decision:

| Metric | Value |
|---|---|
| Competition metric (weighted F1) | **0.8712** |
| Constant-`Dead` baseline | 0.7529 |
| Improvement over baseline | +0.1182 (95% CI [0.1049, 0.1324]) |
| AUC | 0.8977 |
| Log loss | 0.2802 |
| Brier score | 0.0858 |
| F1 (Dead) / F1 (Alive) | 0.9259 / 0.6041 |
| Precision / Recall | 0.9125 / 0.9398 |

Development-set cross-validated weighted F1 was 0.8802, vs. 0.8712 on the audit set — a
+0.0090 "selection optimism" gap, which is the honest cost of everything that was tuned on
development data.

**How much of this is real medicine?** Ablating the follow-up-time features drops AUC
measurably, and AUC computed *within* each diagnosis year (holding follow-up length fixed) is
lower than the pooled figure. Both numbers are legitimate for the competition — the test set
carries the same registry structure — but only the within-year figure reflects genuine
prognostic discrimination; the gap is a censoring artifact. See §7 of the notebook for the
exact decomposition.

---

## Data

The `train.csv` / `test.csv` files are the datathon's SEER-derived lung cancer extract and are
**not included in this repository** (competition data-redistribution terms). To reproduce:

1. Obtain `train.csv` and `test.csv` from the IASDS Datathon 2026 competition page.
2. Place them in a directory and point `DATA_DIR` (top of the notebook, §1) at it.
3. Run the notebook top to bottom.

## Reproducing the submission

```bash
pip install -r requirements.txt   # see below
jupyter nbconvert --to notebook --execute team_ABC_notebook.ipynb
```

- Runtime: ~67 minutes on a Kaggle-spec CPU kernel.
- Fully deterministic: fixed `SEED`, fixed fold assignments, fixed model seeds. Two
  independent runs produced byte-identical `submission.csv` files.
- Set `QUICK = True` near the top of the notebook to run the full pipeline end-to-end in
  ~15 minutes with reduced search budgets, to sanity-check that everything executes before
  committing to a full run.
- A safety-net `submission.csv` is written early (§5.4b) from a fixed-hyperparameter model, so
  an interrupted run still leaves a valid submission; it is overwritten by the final ensemble
  in §8.3.

### Requirements

```
pandas
numpy
scikit-learn
lightgbm
catboost
xgboost
optuna
scipy
matplotlib
```

---

## Disclosures

- **Threshold selection.** The final decision threshold was chosen using the public-leaderboard
  scores of three candidate cut-points from the team's own submissions. This is the one place
  the pipeline uses leaderboard feedback; it's stated explicitly at the `SUBMIT_DEAD_RATE`
  definition in §8.3, alongside the purely out-of-fold alternative (set `SUBMIT_DEAD_RATE = None`
  to use it instead). Nothing else in the pipeline uses leaderboard information, and no
  test-set label was inspected, annotated, reconstructed, or assigned at any point.
- **No AutoML** — every model, feature, constraint, and hyperparameter search space is
  hand-specified in the notebook. Optuna is used only as a numerical optimizer over those
  spaces.
- **No external code or analysis** was taken from another team.

## Limitations

Stated in full in §9 of the notebook; summarized here:

1. **The strongest predictor is a registry artifact, not medicine.** Follow-up length
   dominates the model's discrimination. That's fine for this leaderboard (train and test
   share the same structure — confirmed by adversarial validation in §2.3) but would be
   worthless clinically: you can't improve a patient's prognosis by diagnosing them earlier in
   the observation window.
2. **`vital_status` is all-cause mortality**, not cancer-specific — part of what looks like
   signal is just background mortality risk.
3. **The performance ceiling is set by variables not in the data.** §2.5 shows patients
   identical on every recorded field still differ in outcome; comorbidity, performance status,
   smoking history, and molecular markers (EGFR, ALK, PD-L1) aren't in this extract, and no
   model can recover them.
4. **Treatment variables are confounded by indication** — e.g., `had_surgery` looks
   protective partly because healthier, earlier-stage patients are the ones selected for
   surgery. Not a causal effect.
5. **Threshold transfer assumes the random-split structure holds** — under any temporal or
   institutional shift, the tuned cut-point would need revisiting (though the F1 surface is
   flat enough near the optimum that the risk is small).
6. **One audit set, one estimate** — the reported audit score has the sampling variance of
   4,800 rows; nested cross-validation would tighten this at roughly 10× the compute cost.

## License

_Add your team's / institution's license here._

## Team

Team ABC — IASDS Datathon 2026 Online Qualifier.
