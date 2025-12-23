import os
import mlflow
import mlflow.sklearn
import pandas as pd
import argparse


EXPERIMENT_NAME = "customer-churn"


def get_latest_run_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
        output_format="pandas",
    )

    if runs.empty:
        raise RuntimeError("No runs found in experiment")

    return runs.iloc[0]["run_id"]


def main():
    parser = argparse.ArgumentParser("Batch test churn predictions")
    parser.add_argument("--csv", default="data/test_customers.csv",
                        help="CSV file with customer data to predict")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # Load model
    run_id = get_latest_run_id(EXPERIMENT_NAME)
    model_uri = f"runs:/{run_id}/model"
    print(f"📦 Loading model from run: {run_id}")
    model = mlflow.sklearn.load_model(model_uri)

    # Load batch customer data
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # Keep customer_id for reporting
    if "customer_id" in df.columns:
        customer_ids = df["customer_id"].tolist()
        df = df.drop(columns=["customer_id"])
    else:
        customer_ids = [f"customer_{i+1}" for i in range(len(df))]

    # One-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with model
    model_features = model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    # Predictions
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    # Display results
    print("\n🧪 Batch Test Results")
    print("------------------------")
    for cid, pred, prob in zip(customer_ids, predictions, probabilities):
        print(f"{cid}: Churn Prediction = {'YES' if pred==1 else 'NO'}, "
              f"Probability = {prob:.2%}")


if __name__ == "__main__":
    main()

