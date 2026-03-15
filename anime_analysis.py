import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
anime_df = pd.read_csv('anime.csv')
rating_df = pd.read_csv('rating.csv')

# Basic exploration
print("Anime Dataset Shape:", anime_df.shape)
print("Rating Dataset Shape:", rating_df.shape)
print("\n--- Anime Dataset ---")
print(anime_df.head())
print("\n--- Anime Info ---")
print(anime_df.info())
print("\n--- Missing Values ---")
print(anime_df.isnull().sum())

# Drop rows with missing values
anime_df = anime_df.dropna()

# Clean HTML entities in name column (like &#039; we saw in Gintama)
anime_df['name'] = anime_df['name'].str.replace('&#039;', "'")
anime_df['name'] = anime_df['name'].str.replace('&amp;', '&')

# Convert episodes to numeric (it's currently a string)
anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce')

# Remove -1 ratings from rating_df (means user watched but didn't rate)
rating_df = rating_df[rating_df['rating'] != -1]

# Confirm cleaning
print("Cleaned Anime Dataset Shape:", anime_df.shape)
print("Cleaned Rating Dataset Shape:", rating_df.shape)
print("\nMissing values after cleaning:")
print(anime_df.isnull().sum())

# Fill missing episodes with 0
anime_df['episodes'] = anime_df['episodes'].fillna(0).astype(int)

print("Missing values after final cleaning:")
print(anime_df.isnull().sum())

# ---- ANALYSIS ----

# 1. Top 10 highest rated anime
print("\n--- Top 10 Highest Rated Anime ---")
top_rated = anime_df.nlargest(10, 'rating')[['name', 'rating', 'members']]
print(top_rated.to_string(index=False))

# 2. Top 10 most popular anime (by members)
print("\n--- Top 10 Most Popular Anime ---")
top_popular = anime_df.nlargest(10, 'members')[['name', 'rating', 'members']]
print(top_popular.to_string(index=False))

# 3. Anime type distribution
print("\n--- Anime by Type ---")
print(anime_df['type'].value_counts())

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Top 10 Most Popular Anime (by members)
plt.figure()
top_popular = anime_df.nlargest(10, 'members')
sns.barplot(data=top_popular, x='members', y='name', palette='viridis')
plt.title('Top 10 Most Popular Anime by Members')
plt.xlabel('Number of Members')
plt.ylabel('Anime')
plt.tight_layout()
plt.savefig('top_popular.png')
plt.show()

# 2. Anime Type Distribution
plt.figure()
type_counts = anime_df['type'].value_counts()
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(type_counts)))
plt.title('Anime Type Distribution')
plt.tight_layout()
plt.savefig('type_distribution.png')
plt.show()

# 3. Rating Distribution
plt.figure()
sns.histplot(anime_df['rating'], bins=20, kde=True, color='purple')
plt.title('Anime Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.show()

# 4. Top 10 Most Common Genres
plt.figure()
genres = anime_df['genre'].str.split(', ').explode()
top_genres = genres.value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title('Top 10 Most Common Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig('top_genres.png')
plt.show()