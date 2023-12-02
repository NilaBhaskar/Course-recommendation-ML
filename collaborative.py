import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

# Function to load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv(r'C:/Users/91979/OneDrive/OneDrive - Amrita Vishwa Vidyapeetham/Desktop/ml_project/Coursera.csv')
    data.drop_duplicates(inplace=True)
    data = data[data['Difficulty Level'] != 'Not Calibrated']
    data = data[data['Course Rating'] != 'Not Calibrated']
    data['Course Rating'] = data['Course Rating'].apply(lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)
    data = data.sample(frac=1).reset_index(drop=True)
    return data

# Function to create TF-IDF matrix
def create_tfidf_matrix(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Course Name'] + ' ' + data['Course Description'] + ' ' + data['Skills'])
    return tfidf_vectorizer, tfidf_matrix

# Function to recommend courses
def recommend_courses(tfidf_vectorizer, tfidf_matrix, desc, top_n=10):
    description = ' '.join(desc)
    tfidf_desc = tfidf_vectorizer.transform([description])
    sim_score = list(enumerate(linear_kernel(tfidf_desc, tfidf_matrix)[0]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)

    crs_ind = [i[0] for i in sim_score[1:top_n + 1]]
    recommended_courses = data.iloc[crs_ind]
    recommended_courses = recommended_courses[recommended_courses['Course Rating'] > 3.5]
    
    return recommended_courses

# Load data
data = load_data()
tfidf_vectorizer, tfidf_matrix = create_tfidf_matrix(data)

# Streamlit UI
st.title('Coursera Course Recommender')

user_input = st.text_input("Enter keywords for course recommendation (separated by spaces):")
num_recommendations_val = st.slider("Number of recommendations", 1, 10, 5)
if st.button('Recommend Courses'):
    if user_input:
        keywords = user_input.split()
        recommendations = recommend_courses(tfidf_vectorizer, tfidf_matrix, keywords)
        if recommendations.empty:
            st.write("No courses found based on the keywords.")
        else:
            st.write("Recommended courses based on keywords:")
            st.dataframe(recommendations.head(num_recommendations_val))
    else:
        st.write("Please enter some keywords.")
