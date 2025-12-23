import os
import mlflow
import mlflow.sklearn
import pandas as pd


EXPERIMENT_NAME = "customer-churn"


def get_latest_run_id(experiment_name: str) -> str:
    """
    Fetch the most recent MLflow run ID from an experiment.
    """
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
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    run_id = get_latest_run_id(EXPERIMENT_NAME)
    model_uri = f"runs:/{run_id}/model"

    print(f"📦 Loading model from run: {run_id}")

    model = mlflow.sklearn.load_model(model_uri)

    # -----------------------------
    # Sample test customer
    # -----------------------------
    test_customer = {
        "age": 42,
        "tenure_months": 14,
        "monthly_charges": 72.5,
        "total_charges": 1015.0,
        "support_tickets": 1,
        "gender": "Female",
        "contract_type": "One year",
        "payment_method": "Credit Card",
    }

    df = pd.DataFrame([test_customer])

    # Apply same encoding as training
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with model expectations
    model_features = model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    print("\n🧪 Test Result")
    print("------------------------")
    print(f"Churn Prediction : {'YES' if prediction == 1 else 'NO'}")
    print(f"Churn Probability: {probability:.2%}")


if __name__ == "__main__":
    main()

