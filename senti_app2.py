import streamlit as st
import pickle
import pandas as pd

# Function to load models
def load_models():
    # Load the vectorizer.
    with open('vectoriser-ngram-(1,2).pickle', 'rb') as file:
        vectoriser = pickle.load(file)

    # Load the bagging Model.
    with open('Sentiment-Bagging-SVM.pickle', 'rb') as file:
        bagging_model = pickle.load(file)

    return vectoriser, bagging_model

# Function to preprocess text
def preprocess(text):
    # Add your text preprocessing logic here
    return text

# Function to predict sentiment
def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform([preprocess(text)])
    sentiment = model.predict(textdata)

    # Convert the result into a Pandas DataFrame.
    if sentiment[0] == 0:
        df = "Negative"
    else:
        df = "Positive"
    return df

def main():
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon=":satisfied:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Sentiment Analysis using SVM Model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)


    # Loading the models.
    vectoriser, bagging_model = load_models()

    # Text input for sentiment analysis with native Streamlit styling
    text_input = st.text_area("Enter Text", "Type Here", height=150)

    if st.button("Analyze Sentiment"):
        # Predict sentiment
        result_df = predict(vectoriser, bagging_model, text_input)
        st.success(f"Sentiment: {result_df}")

    # Add a footer
    st.markdown(
        """
        ---
        #### About
        This web app performs sentiment analysis using an SVM model. The SVM model, a powerful classification algorithm, is trained on the preprocessed data to predict sentiment labels (positive or negative). Bagging, or Bootstrap Aggregating, is then applied to enhance the model's performance. Multiple subsets of the training data are sampled with replacement, and individual SVM models are trained on each subset. 
        """
    )

if __name__ == "__main__":
    main()