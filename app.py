from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd 
import pickle
import joblib
import pickle

filename = 'spam_detector.pkl'
clf = joblib.load(open(filename, 'rb'))
cv= joblib.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form["message"]
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)

@app.route('/',methods=['POST'])
def back():
    if request.method == 'POST':
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)