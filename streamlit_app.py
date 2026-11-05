"""
CHURN AI — Automatic ML Pipeline Monitor

Run:
    streamlit run streamlit_app_premium.py

This dashboard monitors the existing automatic pipeline.
It does not independently simulate data or retrain models.
"""

import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

INCOMING_FILE = ROOT / "incoming_customers.csv"
REFERENCE_FILE = ROOT / "cleaned_churn.csv"
MODEL_FILE = ROOT / "churn_model.joblib"
TEST_FILE = ROOT / "test_data.joblib"

STATUS_FILE = ROOT / "pipeline_status.json"
EVENTS_FILE = ROOT / "pipeline_events.jsonl"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Churn AI | Pipeline Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM DASHBOARD DESIGN
# =========================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap'
);

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: radial-gradient(
        ellipse at 15% 0%,
        #172647 0%,
        #0b1020 50%
    );
    color: #f0f3ff;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #0a1020;
    border-right: 1px solid #202d45;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #151f35,
        #10182a
    );
    border: 1px solid #293750;
    padding: 14px 16px;
    border-radius: 14px;
    min-height: 115px;
}

[data-testid="stMetricLabel"] p {
    color: #8fa0bd !important;
    font-size: 0.8rem;
}

[data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 1.5vw, 1.7rem);
    overflow-wrap: anywhere;
    word-break: normal;
}

.hero {
    background: linear-gradient(
        110deg,
        rgba(57,135,229,.14),
        rgba(144,133,233,.08)
    );
    border: 1px solid #2a4260;
    border-radius: 18px;
    padding: 20px 25px;
    margin-bottom: 16px;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    color: #f5f7ff;
    margin: 4px 0;
    font-size: 1.8rem;
}

.hero p, .muted {
    color: #9eb0cc;
}

.eyebrow {
    color: #6fb1ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: .72rem;
    font-weight: 700;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 18px;
    margin-bottom: 10px;
    color: #eef3ff;
}

.status-box {
    background: #111c30;
    border: 1px solid #293750;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# READ STATUS FILE
# =========================================================

def load_status():

    if not STATUS_FILE.exists():
        return {}

    try:

        with open(
            STATUS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (OSError, json.JSONDecodeError):

        return {}


# =========================================================
# READ EVENT HISTORY
# =========================================================

def load_events():

    events = []

    if not EVENTS_FILE.exists():
        return events

    try:

        with open(
            EVENTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if not line.strip():
                    continue

                try:
                    events.append(json.loads(line))

                except json.JSONDecodeError:
                    continue

    except OSError:
        pass

    return events


# =========================================================
# READ CSV FILES
# =========================================================

def read_csv(path):

    if not path.exists():
        return None

    try:

        return pd.read_csv(path)

    except Exception as error:

        st.warning(
            f"Could not read {path.name}: {error}"
        )

        return None


# =========================================================
# CALCULATE MODEL PERFORMANCE
# =========================================================

def calculate_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1-score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }


# =========================================================
# FORMAT PIPELINE DECISION
# =========================================================

def format_decision(decision):

    labels = {
        "candidate_promoted": "Promoted",
        "candidate_rejected": "Rejected",
        "keep_current_no_drift": "No Drift",
        "Pipeline running": "Running",
        "Pipeline failed": "Failed",
    }

    return labels.get(
        decision,
        str(decision).replace("_", " ").title()
    )


# =========================================================
# FORMAT PIPELINE TIME
# =========================================================

def format_time(value):

    if not value:
        return "—"

    try:

        parsed = datetime.fromisoformat(
            str(value)
        )

        return parsed.strftime("%H:%M:%S")

    except (ValueError, TypeError):

        return str(value)


# =========================================================
# AUTOMATIC REFRESH
# =========================================================

try:

    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(
        interval=10000,
        key="automatic_refresh"
    )

    refresh_message = (
        "Automatic refresh: every 10 seconds"
    )

except ImportError:

    refresh_message = (
        "Use the Refresh button to update"
    )

    if st.sidebar.button("↻ Refresh now"):

        st.rerun()


# =========================================================
# LOAD PROJECT DATA
# =========================================================

status = load_status()
events = load_events()

reference = read_csv(REFERENCE_FILE)
incoming = read_csv(INCOMING_FILE)


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown(
    '<div class="eyebrow">MLOps · Customer Churn</div>',
    unsafe_allow_html=True
)

st.title("◈ CHURN.AI — Automatic Pipeline Monitor")

st.markdown(
    """
    Monitor incoming customer data, drift detection,
    model performance and automatic retraining decisions.
    """
)

st.markdown("---")
# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## System Status")

    monitor_status = status.get(
        "monitor_status",
        "Waiting for pipeline status"
    )

    st.write(
        "**Monitor status:**",
        monitor_status
    )

    st.write(
        "**Last update:**",
        status.get("updated_at", "Not recorded")
    )

    st.caption(refresh_message)

    st.markdown("---")

    st.markdown("### Project Files")

    project_files = {
        "Incoming data": INCOMING_FILE,
        "Reference data": REFERENCE_FILE,
        "Model artifact": MODEL_FILE,
        "Test data": TEST_FILE,
        "Pipeline status": STATUS_FILE,
        "Event history": EVENTS_FILE
    }

    for name, path in project_files.items():

        if path.exists():

            st.write(f"🟢 {name}")

        else:

            st.write(f"⚪ {name}")


# =========================================================
# PIPELINE STATUS BANNER
# =========================================================

drift = status.get("drift_detected")

decision = status.get(
    "decision",
    "No decision recorded"
)

error = status.get("error")

if error:

    st.error(
        f"Latest pipeline error: {error}"
    )

elif drift is True:
    st.warning(
        "🚨 Drift was detected in the last pipeline run. "
        "This result remains until the next pipeline run."
    )
elif drift is False:

    st.success(
        "✅ No drift detected in the latest recorded pipeline run."
    )

else:

    st.info(
        "Waiting for the pipeline to record its results."
    )


# =========================================================
# KEY PERFORMANCE INDICATORS
# =========================================================

columns = st.columns(
    [1.15, 1, 1, 1, 1.15]
)

# Latest decision
columns[0].metric(
    "Latest Decision",
    format_decision(decision),
    help=decision
)

# Drift detection
columns[1].metric(
    "Drift Detected",
    (
        "Yes" if drift is True
        else "No" if drift is False
        else "—"
    )
)

# Current model F1
current_f1 = status.get("current_f1")

columns[2].metric(
    "Current Model F1",
    (
        f"{current_f1:.3f}"
        if isinstance(current_f1, (int, float))
        else "—"
    )
)

# Candidate model F1
candidate_f1 = status.get("candidate_f1")

columns[3].metric(
    "Candidate Model F1",
    (
        f"{candidate_f1:.3f}"
        if isinstance(candidate_f1, (int, float))
        else "—"
    )
)

# Last pipeline run — display time only
last_run = status.get("last_run_at")

columns[4].metric(
    "Last Pipeline Run",
    format_time(last_run),
    help=(
        f"Full timestamp: {last_run}"
        if last_run
        else "No pipeline run recorded yet."
    )
)


# =========================================================
# LATEST PIPELINE RESULT
# =========================================================

st.markdown(
    '<div class="section-heading">Latest Pipeline Result</div>',
    unsafe_allow_html=True
)

left, right = st.columns(
    [1.15, 1]
)

with left:

    if status:

        fields = [
            ("Last Run At", "last_run_at"),
            ("Finished At", "finished_at"),
            ("Drifted Features", "drifted_features"),
            ("Decision", "decision"),
            ("Reason", "reason"),
            ("Return Code", "return_code")
        ]

        result_rows = []

        for label, key in fields:

            value = status.get(key, "—")

            if value is None:
                value = "—"

            if key == "decision":
                value = format_decision(value)

            result_rows.append({
                "Field": label,
                "Value": str(value)
            })

        result_df = pd.DataFrame(
            result_rows
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
            height=265
        )

    else:

        st.info(
            "pipeline_status.json has not been created yet. "
            "The pipeline scripts must save their results "
            "before the dashboard can display them."
        )


# =========================================================
# DATASET INFORMATION
# =========================================================

with right:

    dataset_rows = []

    datasets = [
        ("Reference Data", REFERENCE_FILE, reference),
        ("Incoming Data", INCOMING_FILE, incoming)
    ]

    for name, path, df in datasets:

        dataset_rows.append({

            "Dataset": name,

            "Available": (
                "Yes" if path.exists() else "No"
            ),

            "Rows": (
                f"{len(df):,}"
                if df is not None
                else "—"
            ),

            "Columns": (
                len(df.columns)
                if df is not None
                else "—"
            ),

            "Last Modified": (
                datetime.fromtimestamp(
                    path.stat().st_mtime
                ).strftime("%d %b %H:%M")
                if path.exists()
                else "—"
            )
        })

    st.dataframe(
        pd.DataFrame(dataset_rows),
        use_container_width=True,
        hide_index=True,
        height=265
    )


# =========================================================
# INCOMING DATA OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-heading">Incoming Data Overview</div>',
    unsafe_allow_html=True
)

if incoming is None:

    st.info(
        "incoming_customers.csv was not found."
    )

else:

    columns = st.columns(3)

    columns[0].metric(
        "Incoming Rows",
        f"{len(incoming):,}"
    )

    columns[1].metric(
        "Number of Columns",
        len(incoming.columns)
    )

    if "Churn" in incoming.columns:

        churn_rate = (
            incoming["Churn"]
            .astype(str)
            .str.lower()
            .isin(["yes", "1", "true"])
            .mean()
        )

        columns[2].metric(
            "Labelled Churn Rate",
            f"{churn_rate:.1%}"
        )

    else:

        columns[2].metric(
            "Labelled Churn Rate",
            "Labels unavailable"
        )

    features = [
        feature
        for feature in [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
        if feature in incoming.columns
    ]

    if features:

        comparison_rows = []

        for feature in features:

            if (
                reference is not None
                and feature in reference.columns
            ):

                old_values = pd.to_numeric(
                    reference[feature],
                    errors="coerce"
                ).dropna()

            else:

                old_values = pd.Series(
                    dtype=float
                )

            new_values = pd.to_numeric(
                incoming[feature],
                errors="coerce"
            ).dropna()

            old_mean = (
                float(old_values.mean())
                if len(old_values)
                else None
            )

            new_mean = (
                float(new_values.mean())
                if len(new_values)
                else None
            )

            if (
                old_mean is not None
                and new_mean is not None
                and old_mean != 0
            ):

                change = (
                    (new_mean / old_mean) - 1
                ) * 100

            else:

                change = None

            comparison_rows.append({

                "Feature": feature,

                "Reference Mean": (
                    round(old_mean, 3)
                    if old_mean is not None
                    else None
                ),

                "Incoming Mean": (
                    round(new_mean, 3)
                    if new_mean is not None
                    else None
                ),

                "Mean Change (%)": (
                    round(change, 2)
                    if change is not None
                    else None
                )
            })

        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# RECENT AUTOMATIC PIPELINE ACTIVITY
# =========================================================

st.markdown(
    '<div class="section-heading">Recent Automatic Pipeline Activity</div>',
    unsafe_allow_html=True
)

if events:

    event_df = pd.DataFrame(events)

    # Put important columns first when available.
    preferred_columns = [
        "time",
        "event",
        "message",
        "drift_detected",
        "drifted_features",
        "decision",
        "reason",
        "current_f1",
        "candidate_f1",
        "f1_improvement"
    ]

    ordered_columns = [
        col
        for col in preferred_columns
        if col in event_df.columns
    ]

    ordered_columns += [
        col
        for col in event_df.columns
        if col not in ordered_columns
    ]

    event_df = event_df[ordered_columns]

    st.dataframe(
        event_df.iloc[::-1].head(30),
        use_container_width=True,
        hide_index=True,
        height=350
    )

else:

    st.info(
        "No pipeline activity history is available yet."
    )


# =========================================================
# CURRENT MODEL TEST-SET PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-heading">Current Model Test-Set Performance</div>',
    unsafe_allow_html=True
)

if MODEL_FILE.exists() and TEST_FILE.exists():

    try:

        model = joblib.load(
            MODEL_FILE
        )

        test_data = joblib.load(
            TEST_FILE
        )

        X_test = test_data["X_test"]
        y_test = test_data["y_test"]

        metrics = calculate_metrics(
            model,
            X_test,
            y_test
        )

        columns = st.columns(4)

        for column, (name, value) in zip(
            columns,
            metrics.items()
        ):

            column.metric(
                name,
                f"{value:.3f}"
            )

        st.caption(
            "Metrics are calculated from the current model "
            "and the saved test dataset."
        )

    except Exception as error:

        st.warning(
            f"Could not calculate model metrics: {error}"
        )

else:

    st.info(
        "Model file or test_data.joblib is missing."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "CHURN.AI | Automatic ML Pipeline Monitor | "
    "The original monitor.py and retraining_pipeline.py "
    "remain responsible for monitoring, drift detection, "
    "retraining, model promotion and MLflow logging."
)