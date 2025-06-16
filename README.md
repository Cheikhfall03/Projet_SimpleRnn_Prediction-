# Movie Review Sentiment Analysis with RNN and TensorFlow
Ce projet implémente un modèle de réseau de neurones récurrent (RNN) avec une couche d'embedding pour analyser les sentiments des critiques de films à l'aide du jeu de données IMDB.
L'application Streamlit permet aux utilisateurs de saisir une critique de film et d'obtenir une prédiction sur le sentiment (positif ou négatif) de cette critique.
# Prérequis
Avant d'exécuter ce projet, vous devez vous assurer que vous disposez des dépendances suivantes :

Python (version 3.6 ou supérieure)

TensorFlow (version 2.0 ou supérieure)

Streamlit (version 1.0 ou supérieure)

NumPy et Pandas
# Description du modèle
Le modèle de sentiment utilise le jeu de données IMDB, un ensemble de critiques de films annotées avec des étiquettes de sentiment. Le modèle est un RNN simple avec une couche d'embedding, qui encode les mots des critiques en vecteurs denses.

Pré-traitement des données
Les critiques sont pré-traitées en transformant chaque mot en un indice numérique, puis les critiques sont padées à une longueur fixe de 500 mots pour uniformiser la taille des séquences.

Structure du modèle
Le modèle est composé de trois principales couches :

Embedding Layer : Cette couche transforme chaque indice de mot en un vecteur d'embedding dense.

Rnn Layer (SimpleRNN) : Une couche Rnn permet de traiter les séquences de manière efficace et d'extraire des caractéristiques pertinentes du texte.

Dense Layer : Une couche dense avec une activation sigmoid pour classifier les critiques comme positives ou négatives.
# Structure des fichiers
Voici la structure des fichiers du projet :

movie-review-sentiment-analysis/

├── main.py                # Application Streamlit pour l'interface utilisateur

├── model_review_rnn.h5    # Modèle pré-entraîné sauvegardé

├── requirements.txt       # Liste des dépendances Python

├── README.md             # Documentation du projet

└── projet_simple_rnn.ipynb          # Entrainement du modele
