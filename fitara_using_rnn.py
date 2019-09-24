import pandas as pd
df_extract_combined = pd.read_csv('extract_combined.csv')

df_labels = pd.read_csv('labels.csv')

df_final=pd.merge(df_extract_combined,df_labels,on='document_name')
df_text_data=df_final[['text','is_fitara']]

"""**Text preprocessing**"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

df_text_data['text'] = df_text_data['text'].apply(stopwords)
df_text_data.head(10)

nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

def applyLemmatizer(text):
  text = [lemmatizer.lemmatize(word) for word in text.split()]
  return " ".join(text)

df_text_data['text'] = df_text_data['text'].apply(applyLemmatizer)
df_text_data.head(10)

from nltk.stem.snowball import SnowballStemmer 
ps=SnowballStemmer("english")
def applyStemming(text):
  text = [ps.stem(word) for word in text.split()]
  return " ".join(text)

df_text_data['text'] = df_text_data['text'].apply(applyStemming)
df_text_data.head(10)

import re
#len(df_text_data)
for i in range(len(df_text_data)):
  df_text_data['text'][i] = re.sub('[^a-zA-Z]', ' ', df_text_data['text'][i])

"""lookign at the results of applying above preprocessing steps"""

df_text_data['text'][0]

"""removing the extra whitespaces"""

for i in range(len(df_text_data)):
  df_text_data['text'][i] = re.sub('\s+',' ', df_text_data['text'][i])

df_text_data['text'][0]

"""Looking at the top 10 freqent words in each of the row"""

from collections import Counter
nltk.download('inaugural')
def content_text(text):
    frequence_words = Counter()
    for word in text.split():
        frequence_words.update([word])
    # return a list with top ten most common words from each
    return [k for k,_ in frequence_words.most_common(10)]

print(df_text_data['text'].apply(content_text))

"""removing http www and htm as they are there in the dataset and since yes,no were not removed even after applying stopword removal,removing them also.Also there are single characters in the dataset,so removing them by keeping words with length greater than 2."""

def standardize_text(text):
    #text = text.str.replace(r"http\S+", "")
    text = text.replace(r"http", "")
    text = text.replace(r"www", "")
    text = text.replace(r"htm", "")
    text = text.replace(r"yes", "")
    #df[text_field] = df[text_field].str.replace(r"@\S+", "")
    #df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    #df[text_field] = df[text_field].str.replace(r"@", "at")
    #df[text_field] = df[text_field].str.lower()
    return ' '.join( [w for w in text.split() if len(w)>2] )

df_text_data['text'] = df_text_data['text'].apply(standardize_text)
df_text_data.head(10)

df_text_data['text'][0]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_text_data['is_fitara']=le.fit_transform(df_text_data['is_fitara'])

df_text_data.head(10)

"""checking the number total number of unique words and the number of unique words in each of the row."""

dfset=set(df_text_data['text'])
lenlist=[len(x.split()) for x in dfset]
print(sum(lenlist))
lenlist[:20]

def content_text(text):
    frequence_words = Counter()
    for word in text.split():
        frequence_words.update([word])
    # return a list with top ten most common words from each
    return frequence_words

print(df_text_data['text'].apply(content_text))
# results = Counter()
# df_text_data['text'].str.split().apply(results.update)
# print results

"""looking at the first row of the dataset to check if there is anything unique about the text that is fitara."""

df_text_data['text'][1]

"""looking at the top 20 frequent words"""

occurence_counter_for_column = Counter(df_text_data['text'].to_string().split())
occurence_counter = Counter(df_text_data['text'][1].split())
print(occurence_counter_for_column.most_common(100))

print(type(df_text_data['text'][1]))
print(type(df_text_data['text']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_text_data['text'], df_text_data['is_fitara'], test_size=0.2, random_state=42,shuffle=True)

"""importing,training and pickling models"""
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
max_features = 1000
maxlen = 150
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

pickle.dump(tokenizer,open('tokenizer.pkl','wb'))

sequences = tokenizer.texts_to_sequences(X_train)
padded_sequences = sequence.pad_sequences(sequences,maxlen=maxlen)

from keras import layers
from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN,Dropout,LSTM

model = Sequential()
model.add(Embedding(max_features, 64))
model.add(SimpleRNN(64))
model.add(Dropout(0.5))
model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.fit(padded_sequences,y_train,batch_size=32,epochs=20,validation_split=0.1)
model.save("Fitara.h5")

"""Evaluating the results on testing data."""

test_sequences = tokenizer.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=maxlen)

results = model.evaluate(test_sequences_matrix,y_test)

results[1]

y_pred=model.predict(test_sequences_matrix)
from sklearn.metrics import log_loss
log_loss(y_test,y_pred)
