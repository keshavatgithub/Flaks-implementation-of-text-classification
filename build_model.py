from model import NLPModel
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
from sklearn.preprocessing import LabelEncoder

def applyLemmatizer(text):
  lemmatizer = WordNetLemmatizer()
  text = [lemmatizer.lemmatize(word) for word in text.split()]
  return " ".join(text)

def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

def build_model():
    model = NLPModel()

    # filename = os.path.join(
    #     os.path.dirname(__file__), 'chalicelib', 'all/train.tsv')
    df_extract_combined = pd.read_csv('extract_combined.csv')
    df_labels = pd.read_csv('labels.csv')

    df_final=pd.merge(df_extract_combined,df_labels,on='document_name')
    df_text_data=df_final[['text','is_fitara']]
    
    for i in range(len(df_text_data)):
       df_text_data['text'][i] = re.sub('[^a-zA-Z]', ' ', df_text_data['text'][i])
    
    df_text_data['text'] = df_text_data['text'].apply(applyLemmatizer)

    #df_text_data['text'] = df_text_data['text'].apply(stopwords)
    
    le = LabelEncoder()
    df_text_data['is_fitara']=le.fit_transform(df_text_data['is_fitara'])
    
    model.vectorizer_fit(df_text_data.loc[:, 'text'])
    #print('Vectorizer fit complete')

    X = model.vectorizer_transform(df_text_data.loc[:, 'text'])
    #print('Vectorizer transform complete')
    y = df_text_data.loc[:, 'is_fitara']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    #print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()

    #model.plot_roc(X_test, y_test)


if __name__ == "__main__":
    build_model()
