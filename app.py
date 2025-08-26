import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_table('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies_data = pd.read_table('ml-100k/u.item', sep='|', encoding='latin-1', names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'])
users = pd.read_table('ml-100k/u.user', sep='\t', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
combined = ratings.merge(movies_data, left_on='item_id', right_on='movie_id', how='inner')[['user_id', 'item_id', 'rating', 'title']]

combined = combined.dropna()
user_item_matrix = combined.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
similarity_matrix = cosine_similarity(user_item_matrix)

st.title('Movie Recommendation System')

user_id = st.number_input('Enter User ID (1-943)', min_value=1, max_value=943, value=1)
if st.button('Get Recommendations'):
    user_index = user_item_matrix.index.get_loc(user_id)
    user_similarities = similarity_matrix[user_index]
    similar_users = user_item_matrix.index[similarity_matrix[user_index].argsort()[::-1][1:11]]
    unseen_movies = combined[combined['user_id'] == user_id]['item_id'].drop_duplicates()
    all_movies = combined['item_id'].drop_duplicates()
    recommended_movie_ids = [movie for movie in all_movies if movie not in unseen_movies]

    recommendation_scores = {}
    for movie in recommended_movie_ids:
        movie_ratings = combined[combined['item_id'] == movie][['user_id', 'rating']]
        common_users = movie_ratings[movie_ratings['user_id'].isin(similar_users)]
        if not common_users.empty:
            common_similarities = [user_similarities[user_item_matrix.index.get_loc(u)] for u in common_users['user_id']]
            weighted_score = np.sum(np.array(common_similarities) * common_users['rating'].values)
            recommendation_scores[movie] = weighted_score / np.sum(common_similarities)
        else:
            recommendation_scores[movie] = 0

    top_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    st.write('Top 5 Recommended Movies:')
    for movie_id, score in top_recommendations:
        movie_title = movies_data[movies_data['movie_id'] == movie_id]['title'].values[0]
        st.write(f"- {movie_title} (Score: {score:.2f})")