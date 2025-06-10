import streamlit as st
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

import string # Import the string module
stopwords.words('english')
def transfrom_text(text):
  text=text.lower()
  text = nltk.word_tokenize(text)
  y=[]
  for i in text:
     if i.isalnum():
       y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
      text=y[:]
  y.clear()



  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms=st.text_input("Enter the message ")

if st.button('Predict'):

  #1 preprocess
  transform_sms=transfrom_text(input_sms)
  #2 vectorize
  vector_input=tfidf.transform([transform_sms])
  #3 predict
  result=model.predict(vector_input)[0]
  #4. display
  if result == 1:
    st.header("SPAM")
  else:
    st.header(" NOT SPAM")