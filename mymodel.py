from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, SimpleRNN,Dropout,LSTM
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras.models import load_model

class RNN_LSTM(object):
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=1000)
        self.rnn = load_model(r"C:\Users\kesgupta\Desktop\nlp practice\restAPI\using anjali link\Deployment-flask-master\flask-rest-setup-master\sentiment-clf\lib\models/Fitara.h5")
        self.value=[]

    def create_tokens(self,X):
        tokenized_txt = self.tokenizer.texts_to_sequences(X)
        sequenced = sequence.pad_sequences(tokenized_txt,maxlen=150)
        return sequenced

    def predict_rnn(self,X):
        self.rnn._make_predict_function()
        pred = self.rnn.predict(X)
        return pred    