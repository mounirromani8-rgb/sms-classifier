# sms-classifier
# SMS Spam/Ham Classifier

[![GitHub last commit](https://img.shields.io/github/last-commit/mounirromani8-rgb/sms-classifier)](https://github.com/mounirromani8-rgb/sms-classifier/commits)
[![GitHub contributors](https://img.shields.io/github/contributors/mounirromani8-rgb/sms-classifier)](https://github.com/mounirromani8-rgb/sms-classifier/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/mounirromani8-rgb/sms-classifier)](https://github.com/mounirromani8-rgb/sms-classifier/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/mounirromani8-rgb)](https://github.com/mounirromani8-rgb?tab=followers)

---

## About The Project

Le but de ce projet est de réaliser une **classification de SMS** pour détecter si un message est **spam** ou **ham**. Le projet utilise des techniques de **Machine Learning et NLP** avec une interface web Streamlit.

### Étapes principales :

1. Récolte des données
2. Nettoyage et préparation des données :
   - Supprimer la ponctuation
   - Tokenization
   - Élimination des stop words
   - Stemming / Lemmatization
3. Vectorisation :
   - N-grams
   - TF-IDF
4. Feature Engineering
5. Création des jeux de données d’entraînement et de test
6. Entraînement de modèles ML
7. Évaluation :
   - Cross-Validation (K-fold)
   - Matrice de confusion
   - Métriques : precision, recall, accuracy



## Modèles utilisés

- Naive Bayes (MultinomialNB)
- SVM (optionnel si extension)
- Évaluation via precision, recall et accuracy

---

## Keywords

Machine Learning, NLP, tokenizer, stemming, lemmatization, vectorisation, N-grams, tf-idf, Feature Engineering, Cross-Validation, k-fold, Naive Bayes

---

## Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-FF7F50?style=for-the-badge&logo=nltk&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## Packages & Library

```python
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import streamlit as st


---
## Déploiement / Live Demo

L'application **SMS Spam/Ham Classifier** est disponible en ligne :

[🔗 Tester l'application](https://sms-classifier-1-wzvn.onrender.com/)

Pour lancer l'application en local :

```bash
# Cloner le repo
git clone https://github.com/mounirromani8-rgb/sms-classifier.git
cd sms-classifier

# Créer un environnement virtuel
python -m venv svenv
# Linux / macOS
source svenv/bin/activate
# Windows
.\svenv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application Streamlit
streamlit run src/sms-detector.py
