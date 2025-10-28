import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Nettoie un message texte : minuscules, suppression de ponctuation, liens, emails et stopwords"""
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)      # emails
    text = re.sub(r'http\S+', ' ', text)      # URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # caractères non alphanumériques
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def preprocess_dataset(input_path="data/raw/SMSSpamCollection",
                       output_path="data/processed/cleaned.csv"):
    """Charge, nettoie et sauvegarde le dataset"""
    df = pd.read_csv(input_path, sep="\t", header=None, names=["label", "text"])
    df.drop_duplicates(subset="text", inplace=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset nettoyé et sauvegardé dans {output_path} ({df.shape[0]} lignes)")
    return df

if __name__ == "__main__":
    preprocess_dataset()
