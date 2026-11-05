import pandas as pd

# File paths
REFERENCE_PATH = "cleaned_churn.csv"
OUTPUT_PATH = "incoming_customers.csv"

# Load the cleaned customer data
df = pd.read_csv(REFERENCE_PATH)

# Select a different sample each time the simulator runs
incoming = df.sample(
    n=min(1000, len(df))
).copy()

# Simulate a change in monthly charges
incoming["MonthlyCharges"] = (
    incoming["MonthlyCharges"] * 1.20
)

# Simulate a change in customer tenure
incoming["tenure"] = (
    incoming["tenure"] * 0.80
).clip(lower=0)

# Save incoming customer data
incoming.to_csv(OUTPUT_PATH, index=False)

print("DRIFT SIMULATION COMPLETED")

print("\nOriginal data averages:")
print(df[["MonthlyCharges", "tenure"]].mean())

print("\nIncoming data averages:")
print(incoming[["MonthlyCharges", "tenure"]].mean())

print("\nIncoming dataset shape:", incoming.shape)
print("Saved incoming data to:", OUTPUT_PATH)

