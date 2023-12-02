import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
def load_data():
    data = pd.read_csv(r'C:\Users\91979\OneDrive\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\ml_project\Coursera.csv')
    return data

df = load_data()

# Preprocess DataFrame
df = df[df['Course Rating'] != 'Not Calibrated']
df = df[df['Difficulty Level'] != 'Not Calibrated']

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(filtered_words)

df['Course Description'] = df['Course Description'].apply(preprocess_text)

# Create and fit models
def create_fit_models(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Course Description'])

    knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn_model.fit(tfidf_matrix)

    return tfidf_vectorizer, knn_model

tfidf_vectorizer, knn_model = create_fit_models(df)

# Recommend courses
def recommend_courses(course_description, num_recommendations=5):
    input_tfidf = tfidf_vectorizer.transform([course_description])
    distances, indices = knn_model.kneighbors(input_tfidf, n_neighbors=num_recommendations)
    recommended_courses = df.iloc[indices[0]]
    recommended_tfidf_matrix = tfidf_vectorizer.transform(recommended_courses['Course Description'])
    cosine_similarities = cosine_similarity(input_tfidf, recommended_tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    final_recommendations = recommended_courses.iloc[related_docs_indices]
    final_recommendations['Similarity Score'] = cosine_similarities[related_docs_indices]
    return final_recommendations

# Streamlit UI
st.title("Enhanced Course Recommendation System")

input_course_description = st.text_area("Enter the course description:")
num_recommendations_val = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    try:
        recommended_courses = recommend_courses(input_course_description, num_recommendations_val)
        if recommended_courses.empty:
            st.write("No recommendations found. Try a different description.")
        else:
            st.write(recommended_courses.drop_duplicates().head(num_recommendations_val))
    except Exception as e:
        st.error(f"An error occurred: {e}")
