import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def data_preprocess(text):
    # 1. lower case
    text = text.lower()
    # 2. tokenization
    text = nltk.word_tokenize(text)
    # 3. removing special character
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            y.append(i)
    text = y[:]
    y.clear()
    # 4. Stemming
    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS  Spam Classifier")

sms = st.text_input("Enter the message ")

# 1. preprocess
preprocess_sms = data_preprocess(sms)

if st.button("Predict"):
    # 2. vectorize
    vect_sms = tfidf.transform([preprocess_sms])
    # 3. predict
    result = model.predict(vect_sms)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')
