# LoyaltyLens

**From churn prediction to personalized retention action**
AI-powered customer retention prototype for telecom, designed to integrate with Salesforce.
Built for the Deloitte Interns Hackathon FY27 by Nouhaila Salahmi and Wissal El Abdoussi (INPT).

## What it does

1. Predicts the churn risk of each telecom customer (Scikit-learn).
2. Explains each prediction with SHAP.
3. Suggests a personalized retention action (GenAI component: [status]).
4. Keeps a human in the loop: the retention manager approves every action.

## Project status

| Component | Status |
|---|---|
| Data analysis and preprocessing | Done |
| Churn model | Done |
| SHAP explainability | Done |
| GenAI recommendations | [In progress] |
| Salesforce API integration and automation | Planned |

## Repository structure

```
notebooks/   01 EDA, 02 preprocessing and feature engineering, 03 modelling
models/      trained model, preprocessors, model card
data/raw/    Telco Customer Churn dataset
churn_pipeline.py   [short description]
streamlit_app.py    demo application
config.toml, requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To rebuild the processed data and the model, run the notebooks in order (01, 02, 03).

## Dataset

IBM Telco Customer Churn (public, Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Results

[Model, main metrics (recall, PR-AUC) and 2-3 top SHAP factors]

## Responsible AI

- Predictions are explained with SHAP.
- No customer-facing action without human approval.
- [Sensitive attributes handling, see audit_sensitive files]

## Limitations

Trained on public data; it must be validated on representative operator data before any real use.