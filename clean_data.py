import pandas as pd

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = "cleaned_churn.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove duplicate records
df = df.drop_duplicates()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
df = df.dropna()

# Remove customer ID because it is an identifier,
# not a useful behavioural feature for this baseline
df = df.drop(columns=["customerID"])

# Save the cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("DATA CLEANING COMPLETED")
print("Cleaned shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")

