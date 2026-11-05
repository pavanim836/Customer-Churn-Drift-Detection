import pandas as pd

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Display basic information
print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES AND MISSING VALUES")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGES")
print(df["Churn"].value_counts(normalize=True) * 100)


