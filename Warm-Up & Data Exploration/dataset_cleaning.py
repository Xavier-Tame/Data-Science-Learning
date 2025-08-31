import pandas as pd

#==Step 2==
#Data cleaning

df = pd.read_csv("netflix_titles.csv")

text_cols = ['type', 'director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in', 'description']

for col in text_cols:
    df[col] = df[col].replace(["", "NA", "N/A", "nan", None], pd.NA)
    if col == 'rating':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna("Unknown")

df.to_csv("netflix_titles_clean.csv", index=False)

print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())