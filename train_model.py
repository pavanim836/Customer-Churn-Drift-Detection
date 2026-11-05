import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATA_PATH = "cleaned_churn.csv"
MODEL_PATH = "churn_model.joblib"

# Load cleaned data
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Preprocess numerical features
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocess categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Build complete model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nINITIAL MODEL RESULTS")
print("---------------------")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and test data for later pipeline stages
joblib.dump(model, MODEL_PATH)

joblib.dump(
    {
        "X_test": X_test,
        "y_test": y_test
    },
    "test_data.joblib"
)

print(f"\nModel saved to {MODEL_PATH}")
print("Test data saved to test_data.joblib")



mlflow.set_experiment("Customer-Churn-Prediction")

with mlflow.start_run(run_name="initial_training"):

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric(
        "accuracy",
        accuracy_score(y_test, y_pred)
    )

    mlflow.log_metric(
        "precision",
        precision_score(y_test, y_pred)
    )

    mlflow.log_metric(
        "recall",
        recall_score(y_test, y_pred)
    )

    mlflow.log_metric(
        "f1_score",
        f1_score(y_test, y_pred)
    )

    mlflow.sklearn.log_model(
    model,
    artifact_path="churn_model",
    serialization_format="pickle"
)