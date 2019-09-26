# Flaks-implementation-of-text-classification
This repository trains and saves a NB classifier and a RNN and adds them to production using flask
# Steps to run the code

Run the model.py

Run build_model.py

After this you will see the pickle(.pkl) files of classifier and vectorizer.

Run fitara_using_rnn.py

After this you will see the pickle of tokenizer and .h5 file of the rnn classifier.

open api.py and uncomment the following code snippet
'''
conn = sqlite3.connect('database.db')
conn.execute('CREATE TABLE fitara (Input TEXT, Prediction TEXT)')
conn.close()
'''

Run api.py and once the table is craeted,comment the above code snippet again

Run api.py,it will create a link for you,open that link

Enter some text

Choose on of the options and click predict
