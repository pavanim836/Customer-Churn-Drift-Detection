import pandas as pd
from pathlib import Path

# Project files
ROOT = Path(__file__).resolve().parent

REFERENCE_FILE = ROOT / "cleaned_churn.csv"
INCOMING_FILE = ROOT / "incoming_customers.csv"

# Number of customers to generate
BATCH_SIZE = 1000

def main():

    print("\n========== NO-DRIFT SIMULATION ==========")

    # 1. Load the original reference dataset
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_FILE}"
        )

    reference = pd.read_csv(REFERENCE_FILE)

    if reference.empty:
        raise ValueError("Reference dataset is empty.")

    # 2. Sample customers from the reference dataset.
    # Sampling from the same distribution should produce
    # a batch that resembles the reference data.
    batch_size = min(BATCH_SIZE, len(reference))

    incoming = reference.sample(
        n=batch_size,
        replace=False,
        random_state=None
    ).copy()

    # 3. Save the new batch.
    # monitor.py watches this file for changes.
    incoming.to_csv(
        INCOMING_FILE,
        index=False
    )

    print(f"Reference rows: {len(reference):,}")
    print(f"Generated incoming rows: {len(incoming):,}")
    print(f"Saved incoming data to: {INCOMING_FILE}")

    print("\nNo intentional drift was introduced.")
    print(
        "The monitor should detect the changed CSV and "
        "run the retraining pipeline."
    )

    print(
        "\nNOTE: Statistical drift detection may still "
        "report drift because the sampled batch is smaller "
        "than the reference dataset."
    )

    print("\n========== SIMULATION COMPLETED ==========")


if __name__ == "__main__":
    main()

