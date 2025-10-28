import streamlit as st
import joblib
import pandas as pd
import os

# Chemin absolu relatif à ce fichier
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model_nb.joblib")
tfidf_path = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)


def predict_message(message):
    X = tfidf.transform([message])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # probabilité que ce soit spam
    return "spam" if pred == 1 else "ham", prob

# Titre et description
st.title("📩 Spam/Ham SMS Classifier")
st.write("Entrez un message SMS pour savoir s'il est spam ou ham.")

# Initialiser l'historique dans la session
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Message", "Prediction", "Spam Probability"])

# Input utilisateur
user_input = st.text_area("Tapez votre message ici :")

if st.button("Classer le message"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un message.")
    else:
        result, prob = predict_message(user_input)
        
        # Affichage coloré
        if result == "spam":
            st.error(f"🚫 Ce message est un SPAM ! (probabilité : {prob:.2f})")
        else:
            st.success(f"✅ Ce message est HAM (non spam). (probabilité : {prob:.2f})")
        
        # Ajouter au tableau d'historique
        st.session_state.history.loc[len(st.session_state.history)] = [user_input, result, f"{prob:.2f}"]

# Afficher l'historique
st.subheader("📊 Historique des messages classés")
st.table(st.session_state.history[::-1])  # afficher du plus récent au plus ancien
