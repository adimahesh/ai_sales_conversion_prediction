"""Train and compare Logistic Regression and Random Forest sales models."""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

DATA_PATH = Path("data/sales_leads.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
TARGET = "converted"
CATEGORICAL_COLUMNS = ["lead_source", "industry", "company_size", "region"]
NUMERIC_COLUMNS = [
    "touchpoints", "days_to_contact", "website_visits",
    "email_engagement", "lead_score", "deal_value"
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

def make_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python generate_data.py` first."
        )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)

    required = set(FEATURE_COLUMNS + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = data.drop_duplicates().copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data[data[TARGET].isin([0, 1])].copy()
    if data[TARGET].nunique() != 2:
        raise ValueError("The 'converted' target must contain both 0 and 1.")

    # EDA figures
    plt.figure(figsize=(7, 4))
    sns.countplot(data=data, x=TARGET)
    plt.title("Sales Lead Conversion Outcome (0 = No, 1 = Yes)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "conversion_rate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=data, x="lead_source", y=TARGET, errorbar=None)
    plt.title("Conversion Rate by Lead Source")
    plt.ylabel("Conversion rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "conversion_by_source.png", dpi=150)
    plt.close()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted = {}
    plt.figure(figsize=(7, 5))
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probabilities)
        results[name] = {
            "roc_auc": float(auc),
            "accuracy_at_0_5": float(accuracy_score(y_test, predictions)),
            "precision_at_0_5": float(precision_score(y_test, predictions, zero_division=0)),
            "recall_at_0_5": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_at_0_5": float(f1_score(y_test, predictions, zero_division=0)),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
        fitted[name] = pipeline
        RocCurveDisplay.from_predictions(y_test, probabilities, name=f"{name} (AUC={auc:.3f})")
        print(f"\n{name}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Accuracy at 0.50 threshold: {results[name]['accuracy_at_0_5']:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = fitted[best_name]
    dump(best_model, MODEL_DIR / "best_sales_model.joblib")

    metadata = {
        "best_model": best_name,
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "test_conversion_rate": float(y_test.mean()),
        "metrics": results,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nBest model by ROC-AUC: {best_name}")
    print(f"Saved model to {MODEL_DIR / 'best_sales_model.joblib'}")
    print(f"Saved metrics to {MODEL_DIR / 'model_metadata.json'}")

    # Permutation importance works with the full pipeline and original columns.
    try:
        importance = permutation_importance(
            best_model, X_test, y_test, scoring="roc_auc",
            n_repeats=5, random_state=42, n_jobs=-1
        )
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importance.importances_mean
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    except Exception as exc:
        warnings.warn(f"Could not calculate permutation importance: {exc}")

if __name__ == "__main__":
    main()
