"""Module 1 -- Predictive Maintenance.

Trains Logistic Regression, Decision Tree and Random Forest on the AI4I
sensor data, compares them on accuracy / precision / recall / F1 / ROC-AUC,
and logs everything to MLflow.

Run:  python src/train_maintenance.py
"""

from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src import config, database
except ImportError:
    import config
    import database


def build_models() -> dict:
    """Three models. class_weight='balanced' everywhere because only 3.4% of
    rows are failures -- without it the models simply predict 'no failure'."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def importances(name, model, X_test, y_test) -> pd.DataFrame:
    """Tree models expose feature_importances_; logistic regression gets
    permutation importance so all three are comparable."""
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        r = permutation_importance(
            model, X_test, y_test, n_repeats=10,
            random_state=config.RANDOM_STATE, scoring="roc_auc",
        )
        vals = np.clip(r.importances_mean, 0, None)

    total = vals.sum() or 1.0
    return (
        pd.DataFrame(
            {
                "model": name,
                "feature": config.FEATURES,
                "label": [config.FEATURE_LABELS[f] for f in config.FEATURES],
                "importance": vals / total,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    print("=" * 60)
    print("MODULE 1 -- Predictive Maintenance")
    print("=" * 60)

    df = database.load_sensor_data()
    X = df[config.FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    print(f"train {len(X_train):,} / test {len(X_test):,} "
          f"({y_train.mean():.2%} failures in train)\n")

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    results, imp_frames, roc_data = [], [], {}
    baseline_acc = 1 - y_test.mean()

    for name, model in build_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_param("model", name)
            mlflow.log_param("n_features", len(config.FEATURES))
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_metrics(metrics)

            results.append({"model": name, **metrics})
            imp_frames.append(importances(name, model, X_test, y_test))

            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                              "auc": metrics["roc_auc"]}

            cm = confusion_matrix(y_test, model.predict(X_test))
            joblib.dump(model, config.MODELS / f"{name.lower().replace(' ', '_')}.pkl")

            print(f"{name}")
            print(f"  accuracy {metrics['accuracy']:.3f} | precision "
                  f"{metrics['precision']:.3f} | recall {metrics['recall']:.3f} "
                  f"| F1 {metrics['f1']:.3f} | ROC-AUC {metrics['roc_auc']:.3f}")
            print(f"  confusion matrix (TN {cm[0,0]}, FP {cm[0,1]}, "
                  f"FN {cm[1,0]}, TP {cm[1,1]})")

    res = pd.DataFrame(results)
    res.to_csv(config.DATA_PROCESSED / "model_comparison.csv", index=False)

    imp = pd.concat(imp_frames, ignore_index=True)
    imp.to_csv(config.DATA_PROCESSED / "feature_importance.csv", index=False)

    with open(config.DATA_PROCESSED / "roc_curves.json", "w") as fh:
        json.dump(roc_data, fh)

    best = res.loc[res["f1"].idxmax(), "model"]
    joblib.dump(
        joblib.load(config.MODELS / f"{best.lower().replace(' ', '_')}.pkl"),
        config.MODELS / "best_model.pkl",
    )
    with open(config.MODELS / "best_model.json", "w") as fh:
        json.dump({"model": best, "features": config.FEATURES}, fh, indent=2)

    print(f"\nBaseline (always predict 'no failure') accuracy: {baseline_acc:.3f}")
    print(f"Best model by F1: {best}  -> models/best_model.pkl")

    top = imp[imp["model"] == best].head(5)
    print("\nTop factors affecting machine failure:")
    for _, row in top.iterrows():
        print(f"  {row['label']:<28} {row['importance']:.1%}")

    print("\nDone. Next: python src/train_segmentation.py\n")


if __name__ == "__main__":
    main()
