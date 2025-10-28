from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from features import load_data, vectorize_text, split_data

# Charger les données
df = load_data()
X, y, tfidf = vectorize_text(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Entraîner le modèle
model = MultinomialNB()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Sauvegarder le modèle et le vecteur TF-IDF
joblib.dump(model, "model_nb.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")

print("✅ Modèle et vecteur TF-IDF sauvegardés !")
