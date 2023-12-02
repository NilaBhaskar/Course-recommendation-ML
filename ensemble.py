import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load data
@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    df['Course Rating'] = pd.to_numeric(df['Course Rating'], errors='coerce')
    df.dropna(subset=['Course Rating'], inplace=True)
    df['Content'] = df['Course Description'] + " " + df['Skills']
    df = df[df['Difficulty Level'] != 'Not Calibrated']
    df.drop_duplicates(inplace=True)
    df['User Preference'] = df['Course Rating'] >= 4.0
    return df

# Function to fit models and create TF-IDF matrix
@st.cache_data
def create_models_and_tfidf(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Content'])
    svd = TruncatedSVD(n_components=50)
    course_features = svd.fit_transform(tfidf_matrix)

    X_train, X_test, y_train, y_test = train_test_split(course_features, df['User Preference'], test_size=0.2, random_state=42)
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2 = LogisticRegression()
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    ensemble = VotingClassifier(estimators=[('rf', model1), ('lr', model2)], voting='soft')
    ensemble.fit(X_train, y_train)

    return tfidf_vectorizer, svd, course_features, ensemble

# Function to recommend courses
def recommend_courses(tfidf_vectorizer, svd, course_features, input_description, num_recommendations=5):
    input_tfidf = tfidf_vectorizer.transform([input_description])
    input_reduced = svd.transform(input_tfidf)
    similarity_scores = cosine_similarity(input_reduced, course_features)
    sorted_indices = similarity_scores.argsort()[0][::-1]
    recommended_courses = df[['Course Name', 'University', 'Course URL', 'Course Rating']].iloc[sorted_indices[:num_recommendations]]
    return recommended_courses

# Load data and models
df = load_data("Coursera.csv")
tfidf_vectorizer, svd, course_features, ensemble = create_models_and_tfidf(df)

# Streamlit UI setup
st.title("Coursera Course Recommender System")

input_description = st.text_input("Enter course description or keywords:")
num_recommendations_val = st.slider("Number of recommendations", 1, 10, 5)
if st.button("Recommend Courses"):
    if input_description:
        recommendations = recommend_courses(tfidf_vectorizer, svd, course_features, input_description,num_recommendations_val)
        st.write("Recommended courses:")
        st.dataframe(recommendations.head(num_recommendations_val))
    else:
        st.write("Please enter some keywords or a course description.")
