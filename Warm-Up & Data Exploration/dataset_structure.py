import pandas as pd

# ==Step 1==
# Check dataset structure

df = pd.read_csv("netflix_titles.csv")

print("\n=== Dataset Shape (Rows, Columns) ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Info ===")
df.info()

print("\n=== Descriptive Statistics ===")
print(df.describe())

print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())