from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from model import NLPModel
from mymodel import RNN_LSTM
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN,Dropout,LSTM
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import load_model
app = Flask(__name__)
api = Api(app)

model = NLPModel()
modelrnn=RNN_LSTM()
clf_path = 'FitaraClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)
vec_path = 'CountVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)        
import sqlite3
'''
conn = sqlite3.connect('database.db')

conn.execute('CREATE TABLE fitara (Input TEXT, Prediction TEXT)')
conn.close()
'''

with open("tokenizer.pkl","rb") as f:
    modelrnn.tokenizer=pickle.load(f)
modelrnn.rnn = load_model("Fitara.h5")
global graph
graph = tf.get_default_graph()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/list')
def list():
   con = sqlite3.connect("database.db")
   con.row_factory = sqlite3.Row
   
   cur = con.cursor()
   cur.execute("select * from fitara")
   
   rows = cur.fetchall(); 
   return render_template("list.html",rows = rows)

@app.route('/predictions',methods=['POST'])
def predictions():
    max_len=150
    prediction=[]
    features = request.form.values()
    features=pd.Series(features)
    if 'ML prediction' in request.form:    
        uq_vectorized = model.vectorizer_transform(features)
        prediction = model.predict_nb(uq_vectorized)
    else:
        sequenced=modelrnn.create_tokens(features)
        with graph.as_default():
            prediction = modelrnn.predict_rnn(sequenced)

    # Output either 'Negative' or 'Positive' along with the score
    predicting=prediction[0][1]
    predicting=float(predicting)
    if predicting == 0:
            pred_text = 'is not'
    else:
            pred_text = 'is'

    # round the predict proba value and set to new variable
    #confidence = round(pred_proba[0], 3)
    # create JSON object
    output = pred_text
    input1 = request.form['input']
    with sqlite3.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO fitara (Input,Prediction) VALUES (?,?)",(input1,output) )
    return render_template('index.html', prediction_text='The entered text {} Fitara text.'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    
