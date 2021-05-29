## Analyse des sentiments des tweets sur Twetter
### Introduction
* L’objectif  est de réaliser une analyse exploratoire et visuelle des tweets présents dans notre jeu de données. Dans un second temps, le but sera de parvenir à classifier à l’aide de différents modèles disponibles en Python, les sentiments des tweets selon qu’ils soient plutôt positifs, ou négatifs. Autrement dit réunir sentiment analysis et NLP et on finit par déployer le modéle sous heroku en utilisant Flask.
### Table of contents
* [Description](#Description)
* [Technologies](#technologies)
* [Setup](#setup)
### Description
* Le dossier Templates:contient le fichier HTML et CSS
* Le fichier app.py: contient les étapes de prétraitement et d'apprentissage du module ainsi que le code Flask qui nous a permet de créer notre application qu'on va déployer par la suite
* Le fichier train_E6oV3lV.csv: c'est notre jeu de données
* Le fichier YouTube Spam Collection: est un notebook jupyter qui contient l'analyse des sentiment sur un autre jeu de données pour détécter si un commentaire sur Youtube, est un spam ou non.
### Technologies
* HTML5
* CSS3
* Python
* Flask
* NLP
### Setup
* Afin de déployer le modèle sous heroku, il faut tout d'abord  créer un compte sur heroku ensuite, suivre ces étapes:
 1. heroku login
 2. Création d’une application Heroku : heroku create furniture-prediction-app --region eu
 3.  git init
 4. git add .
 5. git commit –am "initial change"
 6. git push heroku master
