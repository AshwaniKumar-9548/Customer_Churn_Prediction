# Customer Churn Prediction Project

This project predicts customer churn for the IBM Telco Customer Churn dataset and provides a Streamlit app for single-customer and batch scoring.

## Project Files

- `customer_churn_masterslevel.ipynb`: training notebook with data audit, EDA, feature engineering, model comparison, explainability, and artifact saving.
- `app.py`: Streamlit app for prediction, batch scoring, model performance review, and methodology notes.
- `models/`: generated artifacts from the notebook.
- `requirements.txt`: Python dependencies.

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook from top to bottom. This creates or refreshes the `models/` folder.

3. Start the app:

```bash
streamlit run app.py
```

## Modeling Objective

The model estimates the probability that a telecom customer will churn. The app uses that probability to classify customers as low, medium, or high risk and suggests retention actions.

## Data

Each row represents one customer. The target is `Churn`, where `Yes` means the customer left and `No` means the customer stayed.

Feature groups:

- Demographics: gender, senior citizen, partner, dependents.
- Account: tenure, contract, paperless billing, payment method.
- Services: phone, internet, security, backup, protection, tech support, streaming.
- Charges: monthly charges and total charges.

## Recent Improvements

- Fixed gender encoding so `Male` and `Female` are handled correctly.
- Added leakage-safer cross-validation pipelines for SMOTE, scaling, and model fitting.
- Moved ANN threshold tuning away from the test set and onto a validation split.
- Replaced the misleading "DeLong-style" test with a paired bootstrap AUC comparison.
- Added deployment metadata for feature engineering constants.
- Reworked the Streamlit app so single and batch predictions use the same preprocessing logic.
- Made the app load artifacts relative to `app.py`, not the terminal's current directory.
- Added CSV validation and clearer model-performance notes.

## Limitations

This is a public cross-sectional dataset, so the model is predictive but not causal. The retention recommendations are business heuristics and should be validated with real campaign response data before production use.
