# SMS Spam/Ham Classifier

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](LICENSE)

---

## About The Project

Le but de ce projet est de classifier des messages SMS en **spam** ou **ham** en utilisant des techniques de Machine Learning.  

Les principales étapes du projet sont :

1. **Récolte des données** : Dataset `SMSSpamCollection.txt`  
2. **Nettoyage et préparation des données** :  
   - Supprimer la ponctuation  
   - Tokenization  
   - Suppression des stop words  
   - Stemming / Lemmatization  
3. **Vectorisation du texte** :  
   - TF-IDF  
   - N-grams  
4. **Feature Engineering** : enrichir et transformer les données  
5. **Construction des jeux de données** pour entraînement et test  
6. **Entraînement et évaluation des modèles** :  
   - Métriques : Precision, Recall, Accuracy  
   - Matrice de confusion  
7. **Déploiement** : interface web avec Streamlit  

---

## Live Demo

L'application est disponible en ligne :

[🔗 Tester l'application](https://sms-classifier-1-wzvn.onrender.com/)

---

## Dataset

Le dataset **SMSSpamCollection.txt** contient deux informations principales : le contenu d'un SMS et le label `spam` ou `ham`.  

Sources :

1. Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. DOCENG'11, 2011.  
2. Gómez Hidalgo, J.M., Almeida, T.A., Yamakami, A. *On the Validity of a New SMS Spam Collection*. ICMLA'12, 2012.  
3. Almeida, T.A., Gómez Hidalgo, J.M., Silva, T.P. *Towards SMS Spam Filtering: Results under a New Dataset*. IJISS, 2013.  

---

## Installation & Run Locally

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
