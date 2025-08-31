import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#==Step 3 — Exploratory Data Analysis (EDA)==

df=pd.read_csv("netflix_titles_clean.csv")

# Most common genres using Matplotlib
plt.figure(figsize=(12,6))
df['listed_in'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title("Top 10 Genres")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Shows by release year using Seaborn
plt.figure(figsize=(12,6))
sns.countplot(y="release_year", data=df, order=df['release_year'].value_counts().index[:10])
plt.title("Top 10 Release Years")
plt.show()