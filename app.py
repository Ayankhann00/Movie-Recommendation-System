import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

@st.cache_data
def load_data():
    ratings = pd.read_table(
        "ml-100k/u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    movies = pd.read_table(
        "ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        names=["movie_id", "title", "release_date", "video_release_date", "imdb_url",
               "unknown", "action", "adventure", "animation", "children", "comedy",
               "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
               "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western"]
    )
    combined = ratings.merge(movies, left_on="item_id", right_on="movie_id", how="inner")[
        ["user_id", "item_id", "rating", "title"]
    ]
    return combined, movies

combined, movies = load_data()

user_item_matrix = combined.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
similarity_matrix = cosine_similarity(user_item_matrix)

def recommend_movies(target_user, top_n=5):
    user_index = user_item_matrix.index.get_loc(target_user)
    user_similarities = similarity_matrix[user_index]
    similar_users = user_item_matrix.index[similarity_matrix[user_index].argsort()[::-1][1:11]]
    unseen_movies = combined[combined["user_id"] == target_user]["item_id"].drop_duplicates()
    all_movies = combined["item_id"].drop_duplicates()
    recommended_movie_ids = [m for m in all_movies if m not in unseen_movies]
    recommendation_scores = {}
    for movie in recommended_movie_ids:
        movie_ratings = combined[combined["item_id"] == movie][["user_id", "rating"]]
        common_users = movie_ratings[movie_ratings["user_id"].isin(similar_users)]
        if not common_users.empty:
            common_similarities = [user_similarities[user_item_matrix.index.get_loc(u)]
                                   for u in common_users["user_id"]]
            weighted_score = np.sum(np.array(common_similarities) * common_users["rating"].values)
            recommendation_scores[movie] = weighted_score / np.sum(common_similarities)
        else:
            recommendation_scores[movie] = 0
    top_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return movies[movies["movie_id"].isin([m[0] for m in top_recs])][["title"]]

user_id = st.number_input("Enter User ID", min_value=1, max_value=int(combined["user_id"].max()), step=1)
top_n = st.slider("Number of Recommendations", 1, 20, 5)

if st.button("Recommend"):
    recs = recommend_movies(user_id, top_n)
    st.write("### Recommended Movies")
    st.table(recs)

    st.write("### User Similarity Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity_matrix, cmap="YlOrRd", xticklabels=False, yticklabels=False, ax=ax)
    st.pyplot(fig)
