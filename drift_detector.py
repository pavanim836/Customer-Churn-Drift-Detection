import pandas as pd
from scipy.stats import ks_2samp

# Dataset paths
REFERENCE_PATH = "cleaned_churn.csv"
INCOMING_PATH = "incoming_customers.csv"

# Features to monitor
FEATURES_TO_MONITOR = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Significance threshold
P_VALUE_THRESHOLD = 0.05


def detect_drift(reference_data, incoming_data):

    drift_detected = False
    results = {}

    for feature in FEATURES_TO_MONITOR:

        # Extract numerical values
        reference_values = reference_data[feature].dropna()
        incoming_values = incoming_data[feature].dropna()

        # Compare the two distributions
        statistic, p_value = ks_2samp(
            reference_values,
            incoming_values
        )

        # Determine whether significant drift was detected
        feature_drift = p_value < P_VALUE_THRESHOLD

        results[feature] = {
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": feature_drift
        }

        if feature_drift:
            drift_detected = True

    return drift_detected, results


if __name__ == "__main__":

    # Load reference and incoming data
    reference_data = pd.read_csv(REFERENCE_PATH)
    incoming_data = pd.read_csv(INCOMING_PATH)

    # Run drift detection
    drift, results = detect_drift(
        reference_data,
        incoming_data
    )

    print("\n========== DRIFT DETECTION ==========")

    for feature, values in results.items():

        print(f"\nFeature: {feature}")
        print(
            "KS Statistic:",
            round(values["ks_statistic"], 4)
        )
        print(
            "P-value:",
            f"{values['p_value']:.6f}"
        )
        print(
            "Drift Detected:",
            values["drift_detected"]
        )

    print("\n=====================================")

    if drift:
        print("RESULT: DRIFT DETECTED")
        print("Action: Retraining should be considered.")
    else:
        print("RESULT: NO DRIFT DETECTED")
        print("Action: Keep monitoring the data.")