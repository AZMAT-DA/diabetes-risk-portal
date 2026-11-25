# 🩺 Diabetes Risk Assessment Portal

A full-stack machine-learning web app that estimates a person's chance of diabetes from 8 medical measurements, explains the result, saves history per user, and gives an admin an analytics dashboard.

> ⚠️ Educational project only. Trained on the Pima Indians Diabetes dataset (adult women, 21+), so results may not apply to other groups. **Not medical advice.**

**Live demo:** _add your Streamlit link here_

## Features

- **Risk prediction** with Low / Moderate / High tiers (soft-voting ensemble of LightGBM + Random Forest + KNN)
- **"Why this result?" explanation** – a what-if analysis showing how much each input raises or lowers the risk compared with a typical patient
- **Accounts:** patient registration and login (PBKDF2-SHA256 salted password hashing, password rules), plus guest mode
- **Personal history** of past assessments stored in SQLite
- **Admin dashboard** with aggregate analytics (risk distribution, risk by age group, glucose vs risk, assessments per day) and anonymous CSV export
- **Input checks:** warns when medical values are 0 (treated as missing)

## Model results (5-fold cross-validation, 768 patients)

| Metric | Ensemble | Logistic Regression baseline |
|---|---|---|
| Accuracy | 0.762 | 0.773 |
| Recall | 0.605 | 0.571 |
| Precision | 0.679 | 0.730 |
| F1 | 0.639 | 0.637 |
| ROC-AUC | 0.829 | 0.836 |

**Finding:** on this small dataset the simple logistic regression performs about as well as the ensemble. The ensemble has slightly higher recall, which matters for screening, so it is used in the app. Recall is still only about 0.60, so roughly 4 in 10 diabetic patients are missed. Lowering the decision threshold is the next improvement.

## Data cleaning and features

- Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are physiologically impossible, so they are treated as missing and filled with the median.
- Engineered features: Glucose×BMI index, Glucose/Insulin ratio, Age×BMI, Glucose×(1+pedigree).

## Project structure

```
├── main.py            # Streamlit app (UI, login, database, dashboard)
├── train.py           # Trains and evaluates the model, saves model.pkl + metrics.json
├── features.py        # Shared feature engineering
├── make_admin.py      # Promote a registered user to admin
├── diabetes.csv       # Dataset
├── model.pkl          # Trained model
├── metrics.json       # Evaluation results shown in the app
├── diabetes_clean.ipynb  # EDA and model experiments
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python train.py                      # creates model.pkl and metrics.json
python -m streamlit run main.py
```

Create an admin: register in the app, then run `python make_admin.py your_email@example.com`, log out and log in again.

## Limitations

- Small dataset (768 rows) from one population
- The explanation is a what-if analysis, not a causal claim
- SQLite storage is for demo use; a hosted database would be needed for real deployment

## Author

Azmat Mehmood – BS Computer Science, PMAS Arid Agriculture University Rawalpindi