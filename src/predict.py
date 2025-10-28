import joblib

# Charger le modèle et le vecteur TF-IDF
model = joblib.load("model_nb.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

def predict_message(message):
    X = tfidf.transform([message])
    pred = model.predict(X)[0]
    return "spam" if pred == 1 else "ham"

if __name__ == "__main__":
    # Exemple de message à tester
    message = input("Entrez un message SMS à tester : ")
    result = predict_message(message)
    print(f"Résultat : {result}")
