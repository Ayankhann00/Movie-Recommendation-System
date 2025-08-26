import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    ratings = pd.read_table(
        "ml-100k/u.data", 
        sep="\t", 
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    movies_data = pd.read_table(
        "ml-100k/u.item", 
        sep="|", 
        encoding="latin-1",
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
               'unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 
               'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 
               'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
    )
    combined = ratings.merge(
        movies_data, left_on="item_id", right_on="movie_id", how="inner"
    )[['user_id', 'item_id', 'rating', 'title']]
    return ratings, movies_data, combined

ratings, movies_data, combined = load_data()
user_item_matrix = combined.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
similarity_matrix = cosine_similarity(user_item_matrix)

def user_based_recommend(target_user, k=5):
    user_index = user_item_matrix.index.get_loc(target_user)
    user_similarities = similarity_matrix[user_index]
    similar_users = user_item_matrix.index[similarity_matrix[user_index].argsort()[::-1][1:11]]

    unseen_movies = combined[combined["user_id"] == target_user]["item_id"].drop_duplicates()
    all_movies = combined["item_id"].drop_duplicates()
    recommended_movie_ids = [movie for movie in all_movies if movie not in unseen_movies.values]

    recommendation_scores = {}
    for movie in recommended_movie_ids:
        movie_ratings = combined[combined["item_id"] == movie][["user_id", "rating"]]
        common_users = movie_ratings[movie_ratings["user_id"].isin(similar_users)]

        if not common_users.empty:
            common_similarities = [
                user_similarities[user_item_matrix.index.get_loc(u)]
                for u in common_users["user_id"]
            ]
            weighted_score = np.sum(np.array(common_similarities) * common_users["rating"].values)
            recommendation_scores[movie] = weighted_score / np.sum(common_similarities)
        else:
            recommendation_scores[movie] = 0

    top_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return movies_data[movies_data["movie_id"].isin([m[0] for m in top_recommendations])]["title"].tolist()

st.title("🎬 Movie Recommendation System (User-Based)")

st.markdown("This app recommends movies using **user-based collaborative filtering** on the MovieLens 100K dataset.")

user_id = st.number_input("Enter a User ID (1–943)", min_value=1, max_value=943, step=1)
k = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

if st.button("Recommend"):
    recommendations = user_based_recommend(user_id, k)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("✅", movie)

if st.checkbox("Show User Similarity Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="YlOrRd", xticklabels=False, yticklabels=False, ax=ax)
    plt.title("User Similarity Heatmap")
    st.pyplot(fig)
