"""
pipeline_core.py
Shared logic used by the dashboard (streamlit_app.py):
  simulate drift -> detect drift -> retrain candidate -> evaluate -> promote / reject

Your original scripts (simulate_drift.py, drift_detector.py, retraining_pipeline.py,
monitor.py) are untouched and keep working. This module re-implements the same ideas
in a form the dashboard can call step by step and visualise.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT / "cleaned_churn.csv"
INCOMING_FILE = ROOT / "incoming_customers.csv"
MODEL_FILE = ROOT / "churn_model.joblib"
CANDIDATE_FILE = ROOT / "candidate_model.joblib"
TEST_FILE = ROOT / "test_data.joblib"
MODELS_DIR = ROOT / "models"
REGISTRY_FILE = ROOT / "model_registry.json"
EVENT_LOG = ROOT / "monitoring_log.jsonl"
BASELINE_FILE = ROOT / "reference_baseline.csv"   # drift baseline after a promotion (cleaned_churn.csv is never modified)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
P_VALUE_THRESHOLD = 0.05      # same rule as drift_detector.py
PSI_WARN, PSI_ALERT = 0.10, 0.25
MIN_F1_IMPROVEMENT = 0.01     # same rule as retraining_pipeline.py
MAX_REGRESSION = 0.05         # guard-rail on the original test set


# --------------------------------------------------------------------------- io
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_FILE)


def load_drift_reference() -> pd.DataFrame:
    """Baseline the incoming data is compared against: the training data, or the batch adopted at the last promotion."""
    return pd.read_csv(BASELINE_FILE if BASELINE_FILE.exists() else REFERENCE_FILE)


def adopt_baseline(incoming: pd.DataFrame) -> None:
    """After a promotion the recent batch becomes the new 'normal' that future batches are compared with."""
    incoming.to_csv(BASELINE_FILE, index=False)


def load_test():
    t = joblib.load(TEST_FILE)
    return t["X_test"], t["y_test"]


def to_binary(churn: pd.Series) -> pd.Series:
    return churn.map({"No": 0, "Yes": 1})


# ------------------------------------------------------------------- event log
def log_event(kind: str, **payload) -> dict:
    row = {"time": now(), "kind": kind, **payload}
    with open(EVENT_LOG, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
    return row


def read_events() -> pd.DataFrame:
    if not EVENT_LOG.exists():
        return pd.DataFrame()
    rows = []
    for line in EVENT_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return pd.DataFrame(rows)


# -------------------------------------------------------------- model registry
def _read_registry() -> list:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return []


def ensure_registry() -> list:
    """First run: snapshot the model that is currently serving as v1."""
    reg = _read_registry()
    if reg or not MODEL_FILE.exists():
        return reg
    MODELS_DIR.mkdir(exist_ok=True)
    dst = MODELS_DIR / "v1_baseline.joblib"
    shutil.copy(MODEL_FILE, dst)
    X, y = load_test()
    reg = [{"version": 1, "file": dst.name, "created": now(), "trigger": "Initial baseline model",
            "metrics": evaluate(joblib.load(MODEL_FILE), X, y), "active": True}]
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2))
    return reg


def registry() -> list:
    return ensure_registry()


def active_version() -> int:
    reg = registry()
    for r in reg:
        if r.get("active"):
            return r["version"]
    return 1


def register_promotion(model, metrics: dict, trigger: str) -> int:
    reg = ensure_registry()
    v = max(r["version"] for r in reg) + 1
    MODELS_DIR.mkdir(exist_ok=True)
    fname = f"v{v}_{datetime.now():%Y%m%d_%H%M%S}.joblib"
    joblib.dump(model, MODELS_DIR / fname)
    joblib.dump(model, CANDIDATE_FILE)   # same files your pipeline writes
    joblib.dump(model, MODEL_FILE)       # <- the model the FastAPI app serves
    for r in reg:
        r["active"] = False
    reg.append({"version": v, "file": fname, "created": now(), "trigger": trigger,
                "metrics": metrics, "active": True})
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2))
    return v


def reset_demo() -> None:
    """Restore v1, clear history, write a drift-free incoming batch."""
    reg = _read_registry()
    if reg:
        v1 = MODELS_DIR / reg[0]["file"]
        if v1.exists():
            shutil.copy(v1, MODEL_FILE)
            shutil.copy(v1, CANDIDATE_FILE)
        for f in MODELS_DIR.glob("v*.joblib"):
            if f.name != reg[0]["file"]:
                f.unlink()
        reg = [dict(reg[0], active=True)]
        REGISTRY_FILE.write_text(json.dumps(reg, indent=2))
    EVENT_LOG.unlink(missing_ok=True)
    BASELINE_FILE.unlink(missing_ok=True)
    ref = load_reference()
    X_test, _ = load_test()
    ref.drop(index=X_test.index).sample(1000, random_state=7).to_csv(INCOMING_FILE, index=False)


# ------------------------------------------------------------------ simulation
def simulate_incoming(monthly_charge_change=0.20, tenure_change=-0.20, contract_shift=0.0,
                      concept_drift=0.0, n=1000, seed=None) -> pd.DataFrame:
    """
    Build a labelled 'incoming' batch.
      monthly_charge_change / tenure_change : your original simulate_drift.py shifts (+20% / -20%)
      contract_shift : share of long-contract customers moved to Month-to-month (categorical drift)
      concept_drift  : 0..1 - churn BEHAVIOUR changes (price-sensitive month-to-month customers
                       leave far more often, loyal long-contract customers stay). Without this the
                       labels are untouched, so retraining has little to learn.
    Rows are sampled only from the training pool, never from the held-out test set.
    """
    rng = np.random.default_rng(seed)
    ref = load_reference()
    X_test, _ = load_test()
    pool = ref.drop(index=X_test.index)
    inc = pool.sample(n=min(n, len(pool)), random_state=int(rng.integers(1_000_000))).copy()

    inc["MonthlyCharges"] = inc["MonthlyCharges"] * (1 + monthly_charge_change)
    inc["tenure"] = (inc["tenure"] * (1 + tenure_change)).clip(lower=0).round()

    if contract_shift > 0:
        movers = (inc["Contract"] != "Month-to-month") & (rng.random(len(inc)) < contract_shift)
        inc.loc[movers, "Contract"] = "Month-to-month"

    if concept_drift > 0:
        # New churn drivers that the old model never learned (e.g. a competitor targets fibre /
        # e-cheque customers and long-tenure customers), and previously risky short-tenure
        # month-to-month customers on cheap plans now stay.
        fibre_echeck = (inc["InternetService"] == "Fiber optic") & (inc["PaymentMethod"] == "Electronic check")
        long_tenure = inc["tenure"] > 30
        cheap_new = (inc["Contract"] == "Month-to-month") & (inc["MonthlyCharges"] < 75) & (inc["tenure"] < 12)
        u = rng.random(len(inc))
        inc.loc[(fibre_echeck | long_tenure) & (u < 0.75 * concept_drift), "Churn"] = "Yes"
        inc.loc[cheap_new & (u < 0.75 * concept_drift), "Churn"] = "No"
    return inc


# ------------------------------------------------------------------- detection
def psi(reference: pd.Series, incoming: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins. <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    i = np.histogram(incoming, edges)[0] / len(incoming)
    r, i = np.clip(r, 1e-4, None), np.clip(i, 1e-4, None)
    return float(np.sum((i - r) * np.log(i / r)))


def psi_label(v: float) -> str:
    return "Major" if v >= PSI_ALERT else "Moderate" if v >= PSI_WARN else "Stable"


def detect_drift_full(reference: pd.DataFrame, incoming: pd.DataFrame) -> dict:
    num_rows = []
    for f in NUMERIC_FEATURES:
        a = pd.to_numeric(reference[f], errors="coerce").dropna()
        b = pd.to_numeric(incoming[f], errors="coerce").dropna()
        ks, p = ks_2samp(a, b)
        pv = psi(a, b)
        num_rows.append({"Feature": f, "Ref mean": a.mean(), "New mean": b.mean(),
                         "Mean change %": (b.mean() / a.mean() - 1) * 100 if a.mean() else np.nan,
                         "KS statistic": float(ks), "P-value": float(p), "PSI": pv,
                         "PSI level": psi_label(pv), "Drift": bool(p < P_VALUE_THRESHOLD)})
    cat_rows = []
    for f in CATEGORICAL_FEATURES:
        if f not in incoming.columns:
            continue
        cats = sorted(set(reference[f]) | set(incoming[f]))
        table = np.array([[(reference[f] == c).sum() for c in cats],
                          [(incoming[f] == c).sum() for c in cats]])
        _, p, _, _ = chi2_contingency(table)
        r = np.clip(table[0] / table[0].sum(), 1e-4, None)
        i = np.clip(table[1] / table[1].sum(), 1e-4, None)
        pv = float(np.sum((i - r) * np.log(i / r)))
        cat_rows.append({"Feature": f, "Chi-square p-value": float(p), "PSI": pv,
                         "PSI level": psi_label(pv), "Drift": bool(p < P_VALUE_THRESHOLD)})
    num_df, cat_df = pd.DataFrame(num_rows), pd.DataFrame(cat_rows)
    return {
        "numeric": num_df, "categorical": cat_df,
        # decision rule identical to drift_detector.py: any monitored numeric feature with KS p < 0.05
        "drift_detected": bool(num_df["Drift"].any()),
        "n_drifted": int(num_df["Drift"].sum()), "n_features": len(num_df),
        "max_psi": float(max(num_df["PSI"].max(), cat_df["PSI"].max() if len(cat_df) else 0)),
    }


# ------------------------------------------------------------------ evaluation
def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1_score": f1_score(y, pred, zero_division=0)}
    try:
        out["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    except Exception:
        pass
    return {k: float(v) for k, v in out.items()}


def retrain_and_compare(champion, incoming: pd.DataFrame, strategy: str = "combined",
                        decision_basis: str = "new_data", seed: int = 42) -> dict:
    """
    strategy        'incoming'  -> fit on incoming data only (what retraining_pipeline.py does)
                    'combined'  -> fit on original training pool + incoming (recommended)
    decision_basis  'new_data'  -> promote if F1 improves on a held-out slice of the NEW data
                                   AND does not collapse on the original test set
                    'original'  -> promote only if F1 improves on the original test set
                                   (exactly the rule in retraining_pipeline.py)
    """
    X_old, y_old = load_test()
    ref = load_reference()
    train_pool = ref.drop(index=X_old.index)

    inc = incoming.copy()
    y_inc = to_binary(inc["Churn"])
    ok = y_inc.notna()
    inc, y_inc = inc.loc[ok], y_inc.loc[ok]
    if y_inc.nunique() < 2:
        raise ValueError("Incoming data must contain both churn classes.")
    X_inc = inc.drop(columns=["Churn"])

    Xi_tr, Xi_ho, yi_tr, yi_ho = train_test_split(X_inc, y_inc, test_size=0.3, random_state=seed, stratify=y_inc)

    if strategy == "incoming":
        X_fit, y_fit = Xi_tr, yi_tr
    else:
        X_fit = pd.concat([train_pool.drop(columns=["Churn"]), Xi_tr], ignore_index=True)
        y_fit = pd.concat([to_binary(train_pool["Churn"]), yi_tr], ignore_index=True)

    candidate = clone(champion)
    candidate.fit(X_fit, y_fit)

    res = {
        "champion": {"original_test": evaluate(champion, X_old, y_old), "new_data": evaluate(champion, Xi_ho, yi_ho)},
        "candidate": {"original_test": evaluate(candidate, X_old, y_old), "new_data": evaluate(candidate, Xi_ho, yi_ho)},
        "n_train": int(len(X_fit)), "n_new_holdout": int(len(Xi_ho)), "n_original_test": int(len(X_old)),
        "strategy": strategy, "decision_basis": decision_basis,
    }
    c, k = res["champion"], res["candidate"]
    if decision_basis == "original":
        gain = k["original_test"]["f1_score"] - c["original_test"]["f1_score"]
        promote = gain >= MIN_F1_IMPROVEMENT
        reason = (f"F1 on the original test set changed by {gain:+.4f} (needs ≥ +{MIN_F1_IMPROVEMENT:.2f}).")
    else:
        gain = k["new_data"]["f1_score"] - c["new_data"]["f1_score"]
        regress = c["original_test"]["f1_score"] - k["original_test"]["f1_score"]
        promote = gain >= MIN_F1_IMPROVEMENT and regress <= MAX_REGRESSION
        reason = (f"F1 on the new-data hold-out changed by {gain:+.4f} (needs ≥ +{MIN_F1_IMPROVEMENT:.2f}); "
                  f"drop on the original test set is {max(regress, 0):.4f} (limit {MAX_REGRESSION:.2f}).")
    res.update({"f1_gain": float(gain), "promote": bool(promote), "reason": reason, "candidate_model": candidate})
    return res


# ------------------------------------------------------------------ mlflow log
def log_run_to_mlflow(tracking_uri: str, drift: dict, result: dict | None, decision: str) -> str | None:
    """Best-effort: never lets an MLflow problem break the demo."""
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Customer-Churn-Prediction")
        with mlflow.start_run(run_name="Retraining_Pipeline") as run:
            mlflow.log_param("drift_detected", drift["drift_detected"])
            mlflow.log_param("source", "dashboard")
            for _, r in drift["numeric"].iterrows():
                mlflow.log_metric(f"ks_statistic_{r['Feature']}", float(r["KS statistic"]))
                mlflow.log_metric(f"p_value_{r['Feature']}", float(r["P-value"]))
                mlflow.log_metric(f"psi_{r['Feature']}", float(r["PSI"]))
            if result:
                for k, v in result["champion"]["original_test"].items():
                    mlflow.log_metric(f"current_{k}", v)
                for k, v in result["candidate"]["original_test"].items():
                    mlflow.log_metric(f"candidate_{k}", v)
                for k, v in result["candidate"]["new_data"].items():
                    mlflow.log_metric(f"candidate_newdata_{k}", v)
                for k, v in result["champion"]["new_data"].items():
                    mlflow.log_metric(f"current_newdata_{k}", v)
                mlflow.log_metric("f1_improvement", result["f1_gain"])
                mlflow.log_param("strategy", result["strategy"])
                mlflow.log_param("decision_basis", result["decision_basis"])
            mlflow.set_tag("decision", decision)
            return run.info.run_id
    except Exception:
        return None