import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from drift_detector import detect_drift


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

REFERENCE_PATH = ROOT / "cleaned_churn.csv"
INCOMING_PATH = ROOT / "incoming_customers.csv"
MODEL_PATH = ROOT / "churn_model.joblib"
TEST_DATA_PATH = ROOT / "test_data.joblib"

STATUS_FILE = ROOT / "pipeline_status.json"
EVENTS_FILE = ROOT / "pipeline_events.jsonl"

MIN_F1_IMPROVEMENT = 0.01


# =========================================================
# STATUS AND EVENT LOGGING
# =========================================================

def read_status():

    if not STATUS_FILE.exists():
        return {}

    try:
        return json.loads(
            STATUS_FILE.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError):
        return {}


def update_status(**updates):

    status = read_status()
    status.update(updates)

    status["updated_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    STATUS_FILE.write_text(
        json.dumps(status, indent=4),
        encoding="utf-8"
    )


def log_event(event, **details):

    record = {
        "time": datetime.now().isoformat(
            timespec="seconds"
        ),
        "event": event,
        **details
    }

    with open(
        EVENTS_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(record) + "\n"
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y,
                predictions,
                zero_division=0
            )
        )
    }


# =========================================================
# MAIN RETRAINING PIPELINE
# =========================================================

def main():

    started_at = datetime.now().isoformat(
        timespec="seconds"
    )

    update_status(
        monitor_status="Pipeline running",
        last_run_at=started_at,
        finished_at=None,
        error=None,
        decision="Pipeline running",
        drift_detected=None,
        drifted_features=[],
        current_f1=None,
        candidate_f1=None,
        reason=None,
        return_code=None
    )

    log_event(
        "pipeline_started",
        message="Retraining pipeline started."
    )

    print("\n========== RETRAINING PIPELINE ==========")

    try:

        # -------------------------------------------------
        # 1. LOAD DATA
        # -------------------------------------------------

        reference_data = pd.read_csv(
            REFERENCE_PATH
        )

        incoming_data = pd.read_csv(
            INCOMING_PATH
        )

        # -------------------------------------------------
        # 2. DETECT DRIFT
        # -------------------------------------------------

        drift_detected, drift_results = detect_drift(
            reference_data,
            incoming_data
        )

        print("\nDRIFT RESULTS")

        drifted_features = []

        for feature, result in drift_results.items():

            feature_drift = bool(
                result["drift_detected"]
            )

            if feature_drift:
                drifted_features.append(feature)

            print(
                f"{feature}: "
                f"KS={result['ks_statistic']:.4f}, "
                f"p-value={result['p_value']:.6f}, "
                f"drift={feature_drift}"
            )

        update_status(
            drift_detected=bool(drift_detected),
            drifted_features=drifted_features
        )

        log_event(
            "drift_detection",
            drift_detected=bool(drift_detected),
            drifted_features=drifted_features
        )

        # -------------------------------------------------
        # 3. LOAD CURRENT MODEL
        # -------------------------------------------------

        current_model = joblib.load(
            MODEL_PATH
        )

        # -------------------------------------------------
        # 4. LOAD FIXED TEST DATA
        # -------------------------------------------------

        test_data = joblib.load(
            TEST_DATA_PATH
        )

        X_test = test_data["X_test"]
        y_test = test_data["y_test"]

        current_metrics = evaluate_model(
            current_model,
            X_test,
            y_test
        )

        print("\nCURRENT MODEL METRICS")

        for name, value in current_metrics.items():
            print(f"{name}: {value:.4f}")

        update_status(
            current_f1=current_metrics["f1_score"]
        )

        # -------------------------------------------------
        # 5. CONFIGURE MLFLOW
        # -------------------------------------------------

        mlflow.set_tracking_uri(
            "http://127.0.0.1:5000"
        )

        mlflow.set_experiment(
            "Customer-Churn-Prediction"
        )

        # -------------------------------------------------
        # 6. TRACK PIPELINE EXECUTION
        # -------------------------------------------------

        with mlflow.start_run(
            run_name="Retraining_Pipeline"
        ):

            mlflow.log_param(
                "drift_detected",
                bool(drift_detected)
            )

            for feature, result in drift_results.items():

                mlflow.log_metric(
                    f"ks_statistic_{feature}",
                    float(result["ks_statistic"])
                )

                mlflow.log_metric(
                    f"p_value_{feature}",
                    float(result["p_value"])
                )

            for name, value in current_metrics.items():

                mlflow.log_metric(
                    f"current_{name}",
                    value
                )

            # ---------------------------------------------
            # 7. NO DRIFT: KEEP CURRENT MODEL
            # ---------------------------------------------

            if not drift_detected:

                decision = "keep_current_no_drift"

                reason = (
                    "No data drift detected. "
                    "The existing model remains in service."
                )

                print("\nNO DRIFT DETECTED")
                print("Keeping the existing model.")

                mlflow.set_tag(
                    "decision",
                    decision
                )

                update_status(
                    decision=decision,
                    reason=reason,
                    candidate_f1=None,
                    monitor_status="Monitoring",
                    return_code=0
                )

                log_event(
                    "pipeline_decision",
                    decision=decision,
                    reason=reason,
                    current_f1=current_metrics["f1_score"]
                )

                return

            # ---------------------------------------------
            # 8. PREPARE CANDIDATE TRAINING DATA
            # ---------------------------------------------

            print("\nDRIFT DETECTED")
            print("Training candidate model...")

            if "Churn" not in incoming_data.columns:

                raise ValueError(
                    "Incoming data must contain Churn "
                    "labels for supervised retraining."
                )

            X_incoming = incoming_data.drop(
                columns=["Churn"]
            )

            y_incoming = incoming_data["Churn"].map(
                {"No": 0, "Yes": 1}
            )

            valid_rows = y_incoming.notna()

            X_incoming = X_incoming.loc[
                valid_rows
            ]

            y_incoming = y_incoming.loc[
                valid_rows
            ]

            if y_incoming.nunique() < 2:

                raise ValueError(
                    "Candidate training data must contain "
                    "both churn classes."
                )

            # ---------------------------------------------
            # 9. TRAIN CANDIDATE MODEL
            # ---------------------------------------------

            candidate_model = clone(
                current_model
            )

            candidate_model.fit(
                X_incoming,
                y_incoming
            )

            # ---------------------------------------------
            # 10. EVALUATE CANDIDATE
            # ---------------------------------------------

            candidate_metrics = evaluate_model(
                candidate_model,
                X_test,
                y_test
            )

            print("\nCANDIDATE MODEL METRICS")

            for name, value in candidate_metrics.items():

                print(
                    f"{name}: {value:.4f}"
                )

                mlflow.log_metric(
                    f"candidate_{name}",
                    value
                )

            f1_improvement = (
                candidate_metrics["f1_score"]
                - current_metrics["f1_score"]
            )

            mlflow.log_metric(
                "f1_improvement",
                f1_improvement
            )

            print(
                f"\nF1-score improvement: "
                f"{f1_improvement:.4f}"
            )

            update_status(
                candidate_f1=candidate_metrics["f1_score"]
            )

            # ---------------------------------------------
            # 11. PROMOTE OR REJECT
            # ---------------------------------------------

            if f1_improvement >= MIN_F1_IMPROVEMENT:

                joblib.dump(
                    candidate_model,
                    ROOT / "candidate_model.joblib"
                )

                joblib.dump(
                    candidate_model,
                    MODEL_PATH
                )

                mlflow.log_param(
                    "promotion_threshold",
                    MIN_F1_IMPROVEMENT
                )

                mlflow.set_tag(
                    "decision",
                    "candidate_promoted"
                )

                decision = "candidate_promoted"

                reason = (
                    "Candidate F1-score improved by "
                    f"{f1_improvement:.4f}, meeting the "
                    f"{MIN_F1_IMPROVEMENT:.2f} threshold."
                )

                print(
                    "\nCANDIDATE MODEL PROMOTED!"
                )

            else:

                mlflow.set_tag(
                    "decision",
                    "candidate_rejected"
                )

                decision = "candidate_rejected"

                reason = (
                    "Candidate F1-score improvement "
                    f"({f1_improvement:.4f}) did not meet "
                    f"the {MIN_F1_IMPROVEMENT:.2f} threshold. "
                    "The existing model is retained."
                )

                print(
                    "\nCANDIDATE MODEL NOT PROMOTED."
                )

                print(
                    "Keeping the existing model."
                )

            # ---------------------------------------------
            # 12. SAVE FINAL DECISION
            # ---------------------------------------------

            update_status(
                decision=decision,
                reason=reason,
                current_f1=current_metrics["f1_score"],
                candidate_f1=candidate_metrics["f1_score"],
                monitor_status="Monitoring",
                return_code=0
            )

            log_event(
                "pipeline_decision",
                decision=decision,
                reason=reason,
                current_f1=current_metrics["f1_score"],
                candidate_f1=candidate_metrics["f1_score"],
                f1_improvement=f1_improvement
            )

        print(
            "\n========== PIPELINE COMPLETED =========="
        )

    except Exception as error:

        # Save the error so it appears in the dashboard.
        print(
            f"\nPIPELINE ERROR: {error}"
        )

        update_status(
            monitor_status="Pipeline failed",
            error=str(error),
            return_code=1,
            finished_at=datetime.now().isoformat(
                timespec="seconds"
            )
        )

        log_event(
            "pipeline_error",
            error=str(error)
        )

        # Keep the process exit code nonzero so monitor.py
        # knows that the pipeline failed.
        raise

    finally:

        status = read_status()

        update_status(
            finished_at=datetime.now().isoformat(
                timespec="seconds"
            )
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()

