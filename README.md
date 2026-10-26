# AI-Based Sales Conversion Probability Prediction

A beginner-friendly machine-learning project that estimates the probability that a sales lead will convert. It compares Logistic Regression and Random Forest using ROC-AUC, ranks leads by predicted conversion probability, and provides a Streamlit interface.

> **Data note:** `data/sales_leads.csv` is generated synthetic demo data, not real customer data. Replace it with an approved real dataset before making business decisions.

## Features
- Generates a reproducible demo lead dataset.
- Cleans data and performs exploratory data analysis (EDA).
- Trains Logistic Regression and Random Forest in preprocessing pipelines.
- Compares models with ROC-AUC and selects the better model.
- Saves the selected model and evaluation metadata.
- Scores leads in a Streamlit app and sorts them by conversion probability.
- Shows model-level feature importance and, when available, SHAP explanations.

## 1. Requirements
- Python 3.10–3.12 recommended.
- VS Code.
- Python extension for VS Code.
- Internet access for the first package installation.

## 2. Open the project
Extract the ZIP, then open the `ai_sales_conversion_prediction` folder in VS Code:
**File → Open Folder**.

## 3. Create and activate a virtual environment

Open **Terminal → New Terminal** in VS Code.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal, then activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
```

If Python 3.11 is not installed, use `python -m venv .venv` instead of `py -3.11 -m venv .venv`.

## 4. Generate the demo dataset
```powershell
python generate_data.py
```
This creates `data/sales_leads.csv`.

## 5. Train and evaluate models
```powershell
python train_model.py
```
This creates:
- `models/best_sales_model.joblib`
- `models/model_metadata.json`
- `reports/model_comparison.png`
- `reports/conversion_rate.png`

Check the printed ROC-AUC scores. ROC-AUC measures how well a model ranks converters above non-converters; it is not the same as accuracy.

## 6. Start the Streamlit application
```powershell
streamlit run app.py
```
A browser tab should open automatically. If not, open the local address printed in the terminal (usually `http://localhost:8501`).

## Workflow mapped to the catalog
1. Define the conversion objective.
2. Understand the sales funnel and lead attributes.
3. Load lead data and conversion outcomes.
4. Clean and preprocess the data.
5. Explore conversion patterns with EDA.
6. Create useful lead features.
7. Encode categorical columns and scale numeric columns.
8. Split data into training and test sets.
9. Train Logistic Regression and Random Forest.
10. Compare models using ROC-AUC.
11. Select the best-performing model.
12. Rank leads by conversion probability.
13. Interpret model results for sales prioritization.
14. Save the model.
15. Build a Streamlit lead-scoring UI.
16. Test sample leads and uploaded CSVs.
17. Deploy to Streamlit Community Cloud if desired.
18. Document insights and limitations.

## Use your own dataset
Prepare a CSV with one row per lead and a binary target column named `converted`:
- `1` = converted
- `0` = did not convert

The demo feature columns are:
`lead_source`, `industry`, `company_size`, `region`, `touchpoints`, `days_to_contact`, `website_visits`, `email_engagement`, `lead_score`, `deal_value`.

For training, replace `data/sales_leads.csv` with your dataset or modify `generate_data.py`/`FEATURE_COLUMNS` in `train_model.py` to match your column names. Do not include personally identifiable or confidential customer data unless you have authorization. Avoid target leakage: features must be available at the time a lead is scored.

## Deploy (optional)
1. Push the project to a GitHub repository.
2. Ensure `requirements.txt`, `app.py`, `train_model.py`, and `generate_data.py` are included.
3. In Streamlit Community Cloud, create an app from the repository and set the main file to `app.py`.
4. Because model files are ignored by `.gitignore`, either run training as part of your deployment setup or temporarily include the generated `models/` files in the repository. The easiest beginner route is to run `python generate_data.py` and `python train_model.py` locally, then include `models/best_sales_model.joblib` and `models/model_metadata.json` in the repository. Only commit demo/trained artifacts, never private data.

## Important limitations
- Demo results are illustrative because the dataset is synthetic.
- Predicted probabilities are estimates, not guarantees.
- Choose a decision threshold based on sales capacity and business costs, not only a default 0.50 threshold.
- Evaluate calibration and performance on representative, recent data before real-world use.
