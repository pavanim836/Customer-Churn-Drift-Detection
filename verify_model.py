import joblib
import pandas as pd

# Load the trained model
model = joblib.load("churn_model.joblib")

# Load saved test data
test_data = joblib.load("test_data.joblib")

X_test = test_data["X_test"]
y_test = test_data["y_test"]

print("MODEL LOADED SUCCESSFULLY")
print("Model type:", type(model).__name__)

print("\nTEST DATA LOADED SUCCESSFULLY")
print("Test samples:", len(X_test))
print("Number of features:", X_test.shape[1])

# Predict churn for five customers
predictions = model.predict(X_test.head(5))

print("\nSAMPLE PREDICTIONS")
for i, prediction in enumerate(predictions, start=1):
    label = "Churn" if prediction == 1 else "No Churn"
    print(f"Customer {i}: {label}")

print("\nVerification completed!")