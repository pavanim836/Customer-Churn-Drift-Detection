import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load existing model and test data
model = joblib.load("churn_model.joblib")
test_data = joblib.load("test_data.joblib")

X_test = test_data["X_test"]
y_test = test_data["y_test"]

# Generate predictions
y_pred = model.predict(X_test)

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Customer-Churn-Prediction")

# Log model and metrics
with mlflow.start_run(run_name="Initial_Model"):

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("training_stage", "initial")

    mlflow.log_metric(
        "accuracy",
        accuracy_score(y_test, y_pred)
    )

    mlflow.log_metric(
        "precision",
        precision_score(y_test, y_pred, zero_division=0)
    )

    mlflow.log_metric(
        "recall",
        recall_score(y_test, y_pred, zero_division=0)
    )

    mlflow.log_metric(
        "f1_score",
        f1_score(y_test, y_pred, zero_division=0)
    )

    mlflow.sklearn.log_model(
    sk_model=model,
    name="churn_model",
    serialization_format="cloudpickle"
    )

    print("MODEL LOGGED TO MLFLOW")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(
        y_test, y_pred, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, y_pred, zero_division=0
    ))
    print("F1-score:", f1_score(
        y_test, y_pred, zero_division=0
    ))

print("\nMLflow experiment created successfully!")