from __future__ import annotations
import os
import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser("Customer Churn Prediction with MLflow")

    # Data & experiment
    p.add_argument("--csv", default="data/customers.csv", help="Path to customer CSV file")
    p.add_argument("--target", default="churn", help="Target column (binary)")
    p.add_argument("--experiment", default="customer-churn", help="MLflow experiment name")

    # Run naming
    p.add_argument("--run-base", default="run", help="Base name for MLflow runs (run-1, run-2, ...)")

    # Model params
    p.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    p.add_argument("--max-depth", type=int, default=6, help="Max tree depth")

    # Training params
    p.add_argument("--test-size", type=float, default=0.1, help="Test split ratio")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")

    return p.parse_args()


def get_next_run_name(experiment_name: str, base_name: str) -> str:
    """
    Returns next run name like: run-1, run-2, ...
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return f"{base_name}-1"

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas"
    )

    if runs.empty:
        return f"{base_name}-1"

    # Extract run numbers
    run_numbers = []
    for name in runs["tags.mlflow.runName"].dropna():
        if name.startswith(f"{base_name}-"):
            try:
                run_numbers.append(int(name.split("-")[-1]))
            except ValueError:
                pass

    next_number = max(run_numbers) + 1 if run_numbers else 1
    return f"{base_name}-{next_number}"


def main():
    args = parse_args()

    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = get_next_run_name(args.experiment, args.run_base)

    # Load data
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found")

    # -----------------------------
    # Data cleaning
    # -----------------------------
    df[args.target] = df[args.target].apply(lambda x: 1 if x > 0 else 0)

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # -----------------------------
    # Train & log
    # -----------------------------
    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("run_name", run_name)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, preds, zero_division=0))
        mlflow.log_metric("f1_score", f1_score(y_test, preds, zero_division=0))

        mlflow.sklearn.log_model(model, "model")

        print(f"✅ MLflow run created: {run_name}")


if __name__ == "__main__":
    main()

