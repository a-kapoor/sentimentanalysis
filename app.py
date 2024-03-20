import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit app
st.title('Sentiment Analysis')
user_input = st.text_area('Enter text:', height=200)

if st.button('Analyze Sentiment'):
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)[0]
    sentiment_labels = ['negative', 'positive', 'neutral']
    prediction = sentiment_labels[prediction]
    st.write(f'The sentiment of the provided text is: **{prediction.upper()}**')
