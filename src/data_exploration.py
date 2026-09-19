
 

import pandas as pd
from pathlib import Path

print("Premier League Transfer Fee Predictor")
print("=" * 45)

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the dataset
file_path = project_folder / "premier-league.csv"
df = pd.read_csv(file_path)

print("\nOriginal dataset loaded!")
print("Original rows:", len(df))

# Step 1: Select incoming transfers
df = df[df["transfer_movement"] == "in"].copy()

print("\nAfter selecting incoming transfers:")
print("Rows:", len(df))

# Step 2: Remove rows with missing transfer fees
df = df.dropna(subset=["fee_cleaned"])

print("\nAfter removing missing transfer fees:")
print("Rows:", len(df))

# Step 3: Remove free transfers
df = df[df["fee_cleaned"] > 0].copy()

print("\nAfter removing free transfers:")
print("Rows:", len(df))

# Step 4: Fill missing ages with the median age
df["age"] = df["age"].fillna(df["age"].median())

# Step 5: Remove duplicate rows
df = df.drop_duplicates()

print("\nAfter cleaning:")
print("Rows:", len(df))

# Step 6: Create a data folder
data_folder = project_folder / "data"
data_folder.mkdir(exist_ok=True)

# Step 7: Save the cleaned dataset
output_path = data_folder / "cleaned_transfers.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved file:", output_path)

# Step 8: Display cleaned data
print("\nFirst 5 cleaned rows:")
print(df.head())

print("\nFinal dataset shape:")
print(df.shape)

print("\nData cleaning completed successfully!")