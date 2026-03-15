import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean data
anime_df = pd.read_csv('anime.csv')
anime_df = anime_df.dropna()
anime_df['name'] = anime_df['name'].str.replace('&#039;', "'")
anime_df['name'] = anime_df['name'].str.replace('&amp;', '&')
anime_df = anime_df.reset_index(drop=True)

print("Dataset loaded successfully!")
print("Shape:", anime_df.shape)
print(anime_df[['name', 'genre', 'type', 'rating']].head())
# Combine features into one string for each anime
anime_df['features'] = (
        anime_df['genre'] + ' ' +
        anime_df['type'] + ' '
)

# TF-IDF Vectorizer on features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['features'])

# Compute cosine similarity between all anime
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Similarity matrix shape:", cosine_sim.shape)
print("Recommender built successfully!")

# Create index mapping name to dataframe index
indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()


# Recommendation function
def recommend(anime_name, top_n=5):
    # Check if anime exists
    if anime_name not in indices:
        print(f"'{anime_name}' not found in dataset!")
        return

    # Get index of the anime
    idx = indices[anime_name]

    # Get similarity scores for this anime with all others
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N most similar (skip index 0 which is the anime itself)
    sim_scores = sim_scores[1:top_n + 1]

    # Get anime indices
    anime_indices = [i[0] for i in sim_scores]

    # Return recommended anime
    print(f"\nTop {top_n} Anime similar to '{anime_name}':")
    print("-" * 50)
    result = anime_df[['name', 'genre', 'rating']].iloc[anime_indices]
    for _, row in result.iterrows():
        print(f"📺 {row['name']}")
        print(f"   Genre: {row['genre']}")
        print(f"   Rating: {row['rating']}")
        print()


# Test it!
recommend("Steins;Gate")
recommend("Death Note")

# Interactive mode
print("\n" + "=" * 50)
print("🎌 ANIME RECOMMENDER SYSTEM")
print("=" * 50)

while True:
    user_input = input("\nEnter an anime name (or 'quit' to exit): ").strip()

    if user_input.lower() == 'quit':
        print("Goodbye! 👋")
        break

    try:
        top_n = int(input("How many recommendations? (default 5): ").strip() or 5)
    except ValueError:
        top_n = 5

    recommend(user_input, top_n)