import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from joblib import dump, load

# Function to preprocess data and train model
def preprocess_and_train_model():
    data = pd.read_csv(r"C:\Nila\Coursera.csv")
    data.drop_duplicates(inplace=True)
    data = data[data['Difficulty Level'] != 'Not Calibrated']
    data = data[data['Course Rating'] != 'Not Calibrated']
    data['Course Rating'] = data['Course Rating'].apply(
        lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)
    data['Difficulty Level'] = pd.factorize(data['Difficulty Level'])[0]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    text_features = tfidf_vectorizer.fit_transform(data['Course Description'] + ' ' + data['Skills'])

    kmeans = KMeans(n_clusters=500, init='k-means++', max_iter=900, n_init=10, random_state=0)
    kmeans.fit(text_features)
    dump(kmeans, 'kmeans_model.pkl')  # Save the model

# Uncomment the line below and run it once to train and save your model
# preprocess_and_train_model()

@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Nila\Coursera.csv")
    data.drop_duplicates(inplace=True)
    data = data[data['Difficulty Level'] != 'Not Calibrated']
    data = data[data['Course Rating'] != 'Not Calibrated']
    data['Course Rating'] = data['Course Rating'].apply(
        lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)
    data['Difficulty Level'] = pd.factorize(data['Difficulty Level'])[0]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    text_features = tfidf_vectorizer.fit_transform(data['Course Description'] + ' ' + data['Skills'])

    kmeans = load('kmeans_model.pkl')  # Load the trained model
    data['Cluster'] = kmeans.predict(text_features)

    return data

def recommend_kmeans(course_index, data):
    cluster = data['Cluster'][course_index]
    recommended_courses = data[(data['Cluster'] == cluster) & (data['Course Rating'] > 0)]
    return recommended_courses

# Streamlit interface
st.title('Online Course Recommendation System')

data = load_data()

# User input
user_input = st.text_input("Enter your search query:", "")
num_recommendations_val = st.slider("Number of recommendations", 1, 10, 5)
submit_button = st.button("Submit")

if submit_button and user_input:
    result_index = data[data['Course Name'].str.contains(user_input, case=False)].index
    if not result_index.empty:
        first_index = result_index[0]
        recommendations = recommend_kmeans(first_index, data)
        st.write(recommendations.head(num_recommendations_val))
    else:
        st.write("No matching results found.")
