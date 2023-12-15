import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# Data Loading
movies_df = pd.read_csv("movies_data.csv")
# Data Exploration
movies_df.head()
movies_df.info()
39
# Data Preprocessing
# Handling Missing Values
movies_df.isnull().sum()
# Feature Engineering
# Extracting Features for Movie Similarity
cv = CountVectorizer()
movies_df['combined_features'] = movies_df.apply(lambda row: ' 
'.join([str(row['genres']), str(row['director']), str(row['cast'])]), axis=1)
movie_matrix = cv.fit_transform(movies_df['combined_features'])
# Building Movie Similarity Matrix
cosine_sim = cosine_similarity(movie_matrix)
# Function to Get Movie Recommendations
def get_recommendations(movie_title, cosine_sim=cosine_sim):
 idx = movies_df[movies_df['title'] == movie_title].index[0]
 sim_scores = list(enumerate(cosine_sim[idx]))
 sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
 sim_scores = sim_scores[1:11] # Top 10 similar movies
 movie_indices = [i[0] for i in sim_scores]
 return movies_df.iloc[movie_indices]['title']
# Example Usage
movie_title = "The Dark Knight"
recommended_movies = get_recommendations(movie_title)
print(recommended_movies)
