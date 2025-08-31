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
plt.savefig("top_10_genres.png")
plt.show()

# Shows by release year using Seaborn
plt.figure(figsize=(12,6))
sns.countplot(y="release_year", data=df, order=df['release_year'].value_counts().index[:10])
plt.title("Top 10 Release Years")
plt.savefig("top_10_release_years.png")
plt.show()

sns.set_theme(style="whitegrid")

top_countries = df['country'].value_counts().head(10).reset_index()
top_countries.columns = ['country', 'count']

top_countries['rank'] = range(len(top_countries))

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='country', hue='rank', data=top_countries, palette='viridis', dodge=False)
plt.title("Top 10 Countries Producing Shows & Movies")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig("top_10_countries.png")
plt.show()

# Duration distributions
df['duration_numeric'] = df['duration'].str.extract('(\d+)').astype(float)

# Movies duration
plt.figure(figsize=(12,6))
sns.histplot(df[df['type']=='Movie']['duration_numeric'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("movies_duration_distribution.png")
plt.show()

# Seasons duration
plt.figure(figsize=(12,6))
sns.histplot(df[df['type']=='TV Show']['duration_numeric'], kde=True, bins=30, color='salmon')
plt.title("Distribution of TV Show Seasons and Shows Count")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of Shows")
plt.tight_layout()
plt.savefig("shows_season_distribution.png")
plt.show()

# Highest rating shows by directors (Top 10)
rating_order = df['rating'].value_counts().index

# Calculate director stats: average rating score
# Ratings are ordered (G < PG < PG-13 < R < TV-MA)
top_directors = df.groupby('director')['title'].count().sort_values(ascending=False).head(10).index

plt.figure(figsize=(12,6))
sns.countplot(y='director', data=df[df['director'].isin(top_directors)],
              hue='rating', order=top_directors, palette='coolwarm')
plt.title("Top 10 Directors by Number of Shows & Movies with Ratings")
plt.xlabel("Number of Titles")
plt.ylabel("Director")
plt.legend(title='Rating', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig("top_directors_ratings.png")
plt.show()

#Most popular cast members
# Split cast column and count individual actors/actresses
from collections import Counter
cast_list = df['cast'].dropna().str.split(', ')
cast_counter = Counter([actor for sublist in cast_list for actor in sublist])
top_cast = pd.DataFrame(cast_counter.most_common(10), columns=['actor', 'count'])
top_cast['rank'] = range(len(top_cast))

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='actor', hue='rank', data=top_cast, palette='magma', dodge=False)
plt.title("Top 10 Most Frequent Cast Members")
plt.xlabel("Number of Titles")
plt.ylabel("Actor/Actress")
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig("top_cast_members.png")
plt.show()

print("All plots generated and saved as PNG files!")