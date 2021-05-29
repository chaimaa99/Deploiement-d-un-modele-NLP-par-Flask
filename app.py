from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
app = Flask(__name__)
#Machine Learning code goes here
@app.route('/')
def home():
 return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
 # Importation des données
 df = pd.read_csv("train_E6oV3lV.csv")
 df = df[:10000] #Réduire la dimension en 10000 lignes
 # Features and Labels
 df_x = df['tweet']
 df_y = df.label
 #Prétraitement
 import numpy as np
 def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)
  for i in r:
   input_txt = re.sub(i, '', input_txt)
  return input_txt
 # Removing Twitter Handles (@user)
 df['tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
 # Convertir df_x en liste
 df_x_content = df_x.tolist()
 processed_data_list = []
 for data in df_x_content:
  processed_data = data.lower()
  processed_data = re.sub('[^a-zA-Z]', ' ', processed_data)
  processed_data = re.sub(r'\s+', ' ', processed_data)
  processed_data_list.append(processed_data)
 all_data_words = []
 lemmmatizer = WordNetLemmatizer()
 for data in processed_data_list:
  data1 = data
  words = word_tokenize(data1)
  words = [lemmmatizer.lemmatize(word.lower()) for word in words if
           (not word in set(stopwords.words('english')) and word.isalpha())]
  all_data_words.append(words)
 all_data_words
 # Extract the features with countVectorizer
 all_data_words = pd.Series(all_data_words).astype(str)
 cv = CountVectorizer()
 X = cv.fit_transform(all_data_words)
#Train test split
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
 # Logistic Regression
 model = LogisticRegression()
 model.fit(X_train, y_train)
 model.score(X_test, y_test)
 if request.method == 'POST':4
 comment = request.form['comment']
 data = [comment]
 vect = cv.transform(data).toarray()
 my_prediction = model.predict(vect)
 return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
 app.run(host='127.0.0.1', port=5555, debug=True)