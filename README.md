# 🎌 Anime Analysis & Recommendation System

A data analysis and content-based recommendation system built with Python that analyzes and recommends anime based on genre and type similarities.

> **Note:** This project uses the MyAnimeList dataset collected around 2016, so some ratings and popularity numbers may differ from current MyAnimeList standings.

---

## 📊 Project Overview

This project has two parts:

**Part 1 — Data Analysis** explores a dataset of 12,000+ anime to answer interesting questions like which anime are the most popular, highest rated, and what genres dominate the platform.

**Part 2 — Recommendation System** uses Machine Learning to recommend similar anime based on what you already like, using TF-IDF vectorization and Cosine Similarity.

---

## 📁 Project Structure

```
AnimeR/
├── archive/
│   ├── anime.csv          # Anime dataset (12,294 anime)
│   └── rating.csv         # User ratings (7.8M ratings)
├── anime_analysis.py      # Data analysis & visualizations
└── anime_recommender.py   # ML recommendation system
```

---

## 📈 Analysis Highlights

- **Most Popular Anime:** Death Note with 1M+ members
- **Highest Rated Anime:** Kimi no Na wa, FMA Brotherhood, Steins;Gate
- **Most Common Type:** TV Series (30%)
- **Most Common Genre:** Comedy, Action, Adventure

---

## 🤖 How the Recommender Works

1. **Feature Extraction** — combines each anime's genre and type into a text string
2. **TF-IDF Vectorization** — converts text features into numerical vectors
3. **Cosine Similarity** — measures similarity between all 12,000+ anime
4. **Recommendation** — returns top K most similar anime to your input

---

## 🚀 How to Use

**Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Run the analysis:**
```bash
python anime_analysis.py
```

**Run the recommender:**
```bash
python anime_recommender.py
```

**Example interaction:**
```
🎌 ANIME RECOMMENDER SYSTEM
==================================================
Enter an anime name (or 'quit' to exit): Death Note
How many recommendations? (default 5): 5

Top 5 Anime similar to 'Death Note':
--------------------------------------------------
📺 Death Note Rewrite
   Genre: Mystery, Police, Psychological, Supernatural, Thriller
   Rating: 7.84

📺 Mirai Nikki (TV)
   Genre: Action, Mystery, Psychological, Shounen, Supernatural, Thriller
   Rating: 8.07
```

---

## 📦 Dataset

- **Source:** [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) on Kaggle
- **Collected:** ~2016 (9 years old)
- **Size:** 12,294 anime, 7.8M user ratings
- **Features:** anime_id, name, genre, type, episodes, rating, members

> ⚠️ The dataset CSV files are not included in this repository due to their large size.
> To run this project:
> 1. Download the dataset from the Kaggle link above
> 2. Extract the ZIP file
> 3. Place `anime.csv` and `rating.csv` inside an `archive/` folder in the project directory
```


## 🛠️ Technologies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (TF-IDF, Cosine Similarity)

---

## ⚠️ Limitations
- Dataset is from 2016 so newer anime like Demon Slayer, Jujutsu Kaisen, or Vinland Saga are not included
- Recommendations are based only on genre and type — not story, art style, or mood
- Some obscure anime with very few members appear in top rated due to rating bias
