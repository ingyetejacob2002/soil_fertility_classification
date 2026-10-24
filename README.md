# Soil Fertility Classification System

A machine-learning decision support system that classifies a soil sample into a fertility class
(**Low / Medium / High**) from 12 laboratory soil parameters. Built for the project
*"Development of a Decision Support System for Soil Fertility Classification Using Machine Learning"*.

**Scope:** soil fertility classification only. No crop recommendation, yield prediction or fertilizer advice.

## Dataset

* File: `data/raw/dataset1.csv` - the Kaggle "Soil Fertility Dataset" (Rahul Jaiswal), 880 rows,
  12 chemical features (N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B) and an integer target `Output` (0/1/2).
* Audit (`reports/data_audit.txt`): no missing values, no duplicates. 4 rows removed as implausible
  (pH = 0.9 twice; OC = 24 %; Zn = 38-42 mg/kg with Fe = 34-44 mg/kg), leaving **876** rows.
* Class distribution after cleaning: Low 401, Medium 436, High 39 (strongly imbalanced).

### Dataset limitations (please read, and state them in your thesis)
1. **Label provenance is undocumented.** The file does not say how fertility labels were assigned. The names
   Low/Medium/High (0/1/2) follow published studies that use this dataset; they are not documented in the file.
2. **Units are not documented in the file.** N, P, K, EC, OC units (mg/kg, mg/kg, mg/kg, dS/m, %) follow secondary
   literature; micronutrient units are assumed to be mg/kg.
3. **The labels appear to depend strongly on nitrogen and phosphorus.** A decision tree of depth 3 uses only N and P
   and reaches ~86 % cross-validated accuracy during exploration. This suggests labels may have been derived from
   rules; high accuracy therefore may reflect rule recovery rather than agronomic insight.
4. Single region (not Nigerian soils), chemical parameters only (no texture, bulk density or biological indicators).
5. Only 39 "High" samples (31 in training, 8 in the test set): per-class results for "High" are statistically unstable.

## Results (actual output of `python -m src.train_model` / `python -m src.evaluate_model`)

Split: 700 training / 176 test rows (stratified, seed 42). The model was **selected by 5-fold cross-validated
macro-F1 on the training set only**; the test set was not used for selection.

| Model | CV acc. | CV macro-F1 | Test acc. | Test macro-F1 |
|---|---|---|---|---|
| Baseline (most frequent) | 0.497 | 0.221 | 0.500 | 0.222 |
| Decision stump (depth 1) | 0.487 | 0.357 | 0.472 | 0.350 |
| **Decision Tree (selected)** | 0.889 | **0.781** | 0.835 | 0.759 |
| Random Forest | 0.897 | 0.631 | 0.903 | 0.683 |
| SVM (RBF) | 0.790 | 0.628 | 0.778 | 0.606 |
| K-Nearest Neighbors | 0.754 | 0.516 | 0.830 | 0.566 |
| Naive Bayes | 0.830 | 0.603 | 0.847 | 0.582 |
| Gradient Boosting | 0.897 | 0.759 | 0.864 | 0.679 |
| Neural Network (MLP) | 0.850 | 0.669 | 0.841 | 0.640 |

Selected model, held-out test set: accuracy 0.835, macro precision 0.732, macro recall 0.810, macro-F1 0.759.
Training accuracy is 1.000 (the unpruned tree memorises the training set; judge it by the test results).

**Honest interpretation.** Macro-F1 was chosen as the selection criterion because the classes are imbalanced and
accuracy would favour the majority classes. Random Forest has higher test *accuracy* (0.903) but lower
cross-validated macro-F1 (0.631), mainly because it is weaker on the rare "High" class. With only 8 "High" test
samples the differences between the top models are not statistically conclusive. Chapter Two reports RF/ANN as best
in other studies; on this dataset that was not the case under this criterion.

## Project structure
```
app.py                      Streamlit interface (loads the saved model, never retrains)
src/config.py               paths, feature order, class names, units
src/data_preprocessing.py   load, audit, clean, stratified split, preprocessing pipeline
src/train_model.py          compare 7 algorithms + 2 diagnostics, select, save with joblib
src/evaluate_model.py       held-out evaluation, writes reports/
src/validation.py           input validation
src/prediction.py           artifact loading + prediction
models/                     soil_fertility_model.joblib, feature_schema.json, model_metadata.json
reports/                    data_audit.txt, model_comparison.csv, model_evaluation.txt, metrics.json, confusion_matrix.png
tests/                      unittest/pytest tests
```

## Architecture and prediction workflow
`dataset1.csv` -> `data_preprocessing` (audit, clean, stratified split -> `data/processed/`) -> `train_model`
(cross-validated comparison; best model saved as ONE scikit-learn `Pipeline` = imputer/scaler + classifier, plus
`feature_schema.json` holding feature order, units, observed ranges and input limits) -> `evaluate_model` (reports).

At prediction time: user enters 12 values -> `validation.validate_input` (missing / non-numeric / negative /
out-of-range = error; outside training range = warning) -> values arranged in the schema's feature order ->
the saved Pipeline applies the same preprocessing used in training and predicts -> class label (and the model's
class probabilities where available) shown. The app uses no separate preprocessing code.

## Setup on Windows (PowerShell)
Requires Python 3.11 or newer.
```powershell
# 1. project folder (skip if you already unzipped the project)
mkdir soil_fertility_classification; cd soil_fertility_classification

# 2. virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# if activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. dataset: place dataset1.csv in data\raw\  (already included in this project)

# 5. preprocessing
python -m src.data_preprocessing

# 6. train + compare models, then evaluate
python -m src.train_model
python -m src.evaluate_model

# 7. start the app
streamlit run app.py

# 8. tests
python -m pytest -v
```
The trained model is already included in `models/`; steps 5-6 are only needed to reproduce it.
If `pip` cannot find an exact pinned version for your Python, install the latest versions and re-run steps 5-6
(the saved model must be re-trained if the scikit-learn version differs).

## Verification status
Verified by execution in the authoring environment: preprocessing, training, evaluation, 31 unit tests.
**Not verified there:** the Streamlit app was exercised only with a stubbed Streamlit (all pages, classify,
validation errors, reset, missing-model error) because Streamlit could not be installed; `tests/test_app.py`
(real Streamlit `AppTest`) is skipped unless Streamlit is installed. Run `python -m pytest -v` on your machine to
run it. Screenshots must be taken by you once the app is running.

## Disclaimer
This tool supports preliminary soil fertility classification and does not replace laboratory analysis or
professional agronomic assessment.
