import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import re
import numpy as np

# Function to recommend courses using KNN model and Cosine Similarity
def recommend_knn(course_index, num_recommendations=5):
    # KNN Recommendations
    distances, indices = knn_model.kneighbors(text_features[course_index], n_neighbors=num_recommendations)
    knn_courses = data.iloc[indices[0]]

    # Cosine Similarity Recommendations
    cosine_scores = cosine_sim[course_index]
    cosine_indices = np.argsort(cosine_scores)[::-1][1:num_recommendations+1]
    cosine_courses = data.iloc[cosine_indices]

    # Combine both sets of recommendations
    combined_courses = pd.concat([knn_courses, cosine_courses]).drop_duplicates().head(num_recommendations)
    return combined_courses

# Load and preprocess data
data = pd.read_csv(r"C:\Users\91979\OneDrive\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\ml_project\Coursera.csv")
data.drop_duplicates(inplace=True)
data = data[data['Difficulty Level'] != 'Not Calibrated']
data = data[data['Course Rating'] != 'Not Calibrated']
data['Course Rating'] = data['Course Rating'].apply(lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)
label_encoder = LabelEncoder()
data['Difficulty Level'] = label_encoder.fit_transform(data['Difficulty Level'])

# TF-IDF vectorization and Cosine Similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
text_features = tfidf_vectorizer.fit_transform(data['Course Description'] + ' ' + data['Skills'])
cosine_sim = cosine_similarity(text_features, text_features)

# KNN model
knn_model = NearestNeighbors(n_neighbors=30, metric='cosine')
knn_model.fit(text_features)

# Streamlit app
st.title('Enhanced Course Recommendation App')
user_input = st.text_input("Enter your search query:")
num_recommendations = st.number_input("Number of Recommendations", 1, 10, 5)

# Handle submit button
if st.button("Submit"):
    result_index = data[data['Course Name'].str.contains(user_input, case=False)].index
    if not result_index.empty:
        first_index = result_index[0]
        recommendations = recommend_knn(first_index, num_recommendations)
        df_recommendations = pd.DataFrame(recommendations)
        df_recommendations.sort_values(by='Difficulty Level', ascending=True, inplace=True)
        st.subheader('Recommended Courses:')
        st.table(df_recommendations[['Course Name', 'Course Rating', 'Difficulty Level']])
    else:
        st.warning("No matching results found.")
