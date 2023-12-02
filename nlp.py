import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Function to preprocess and load data
@st.cache_data
def load_data():
    data = pd.read_csv('Coursera.csv')
    data.drop_duplicates(inplace=True)
    data = data[data['Course Rating'] != 'Not Calibrated']
    data = data[data['Difficulty Level'] != 'Not Calibrated']
    data['Course Rating'] = data['Course Rating'].apply(lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)
    return data

# Load data
data = load_data()

# TF-IDF Vectorization with bi-grams
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
features = tfidf.fit_transform(data['Course Description'] + data['Course Name'] + data['Skills'])

# Function to recommend courses
def recommend_courses(user_input):
    preprocessed_input = tfidf.transform([user_input])
    similarity_scores = cosine_similarity(preprocessed_input, features)
    top_indices = similarity_scores.argsort()[0][::-1][:15]
    recommended_courses = data.iloc[top_indices][['Course Name', 'Course Description', 'Difficulty Level', 'Course Rating', 'Skills']]
    return recommended_courses

# Streamlit UI setup
st.title("Coursera Course Recommender")

user_input = st.text_area("Enter course description, name, or skills you're interested in:")
num_recommendations_val = st.slider("Number of recommendations", 1, 10, 5)
if st.button("Recommend Courses"):
    if user_input:
        recommendations = recommend_courses(user_input)
        if recommendations.empty:
            st.write("No courses found based on your input.")
        else:
            st.write("Recommended courses:")
            st.dataframe(recommendations.head(num_recommendations_val))
    else:
        st.write("Please enter some keywords or course details.")
