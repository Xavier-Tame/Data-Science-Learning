import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#==Step 4 — Build a Simple Recommendation System==

df=pd.read_csv("netflix_titles_clean.csv")

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['listed_in'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend similar shows
def recommend(title):
    if title not in df['title'].values:
        return "Show not found."
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [df['title'].iloc[i] for i, _ in sim_scores]
    return recommendations

print(recommend("Blood & Water"))