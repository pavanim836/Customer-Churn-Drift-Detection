import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

INCOMING_FILE = PROJECT_DIR / "incoming_customers.csv"
PIPELINE_FILE = PROJECT_DIR / "retraining_pipeline.py"

STATUS_FILE = PROJECT_DIR / "pipeline_status.json"
EVENTS_FILE = PROJECT_DIR / "pipeline_events.jsonl"

CHECK_INTERVAL = 60


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
        file.write(json.dumps(record) + "\n")


# =========================================================
# FILE CHANGE DETECTION
# =========================================================

def get_file_hash(file_path):
    """Return a SHA-256 hash to identify file changes."""

    if not file_path.exists():
        return None

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(8192), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# RUN THE EXISTING RETRAINING PIPELINE
# =========================================================

def run_retraining_pipeline():

    print("\nStarting drift detection pipeline...")

    update_status(
        monitor_status="Pipeline running",
        error=None,
        return_code=None
    )

    log_event(
        "pipeline_started",
        message="Incoming data changed; pipeline started."
    )

    try:

        result = subprocess.run(
            [sys.executable, str(PIPELINE_FILE)],
            cwd=PROJECT_DIR,
            check=False
        )

        if result.returncode == 0:

            print("Pipeline execution completed.")

            update_status(
                monitor_status="Monitoring",
                return_code=0
            )

            log_event(
                "pipeline_completed",
                message="Pipeline finished successfully."
            )

            return True

        print(
            "Pipeline failed with exit code "
            f"{result.returncode}."
        )

        update_status(
            monitor_status="Pipeline failed",
            error=(
                "Retraining pipeline exited with code "
                f"{result.returncode}. Check the terminal output."
            ),
            return_code=result.returncode
        )

        log_event(
            "pipeline_failed",
            return_code=result.returncode
        )

        return False

    except Exception as error:

        print(f"Error running pipeline: {error}")

        update_status(
            monitor_status="Pipeline failed",
            error=str(error)
        )

        log_event(
            "pipeline_failed",
            error=str(error)
        )

        return False


# =========================================================
# AUTOMATIC MONITOR
# =========================================================

def monitor_incoming_data():

    print("Automatic Drift Monitoring Started")
    print(
        f"Checking for new data every "
        f"{CHECK_INTERVAL} seconds."
    )
    print("Press Ctrl+C to stop.\n")

    last_processed_hash = None

    update_status(
        monitor_status="Monitoring",
        error=None
    )

    log_event(
        "monitor_started",
        message="Automatic monitoring started."
    )

    try:

        while True:

            current_hash = get_file_hash(
                INCOMING_FILE
            )

            if current_hash is None:

                print(
                    "Waiting for incoming_customers.csv..."
                )

                update_status(
                    monitor_status="Waiting for incoming data"
                )

            elif current_hash != last_processed_hash:

                print("New incoming data detected!")

                log_event(
                    "incoming_data_changed",
                    message="A new incoming data file was detected."
                )

                success = run_retraining_pipeline()

                # Mark the file as processed only if the
                # pipeline succeeded and the file did not
                # change while it was being processed.
                if (
                    success
                    and current_hash
                    == get_file_hash(INCOMING_FILE)
                ):
                    last_processed_hash = current_hash

            else:

                print(
                    "No new incoming data. Waiting..."
                )

                update_status(
                    monitor_status="Monitoring"
                )

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:

        print("\nDrift monitoring stopped.")

        update_status(
            monitor_status="Stopped"
        )

        log_event(
            "monitor_stopped",
            message="Monitoring stopped by the user."
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {PIPELINE_FILE}"
        )

    monitor_incoming_data()

