import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data(path="data/processed/cleaned.csv"):
    df = pd.read_csv(path)
    return df

def vectorize_text(df, max_features=3000):
    # Supprimer les lignes où clean_text est NaN
    df = df.dropna(subset=["clean_text"])
    
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df["clean_text"])
    y = df["label"].map({"ham":0, "spam":1})  # 0 = ham, 1 = spam
    return X, y, tfidf


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
"""Module de gestion des fonctionnalités pour le classificateur spam/ham"""