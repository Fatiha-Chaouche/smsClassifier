import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le modèle et le vectoriseur
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Titre de l'application
st.title("SMS Spam Classifier")

# Entrée utilisateur pour le SMS
user_input = st.text_area("Enter the SMS text")

# Fonction pour prédire le SMS
def predict_sms(model, vectorizer, sms):
    sms_vectorized = vectorizer.transform([sms])
    prediction = model.predict(sms_vectorized)
    return "spam" if prediction[0] == 1 else "ham"

# Bouton pour faire la prédiction
if st.button("Classify"):
    if user_input:
        result = predict_sms(model, tfidf, user_input)
        st.write(f'The SMS is classified as: {result}')
    else:
        st.write("Please enter an SMS to classify")
