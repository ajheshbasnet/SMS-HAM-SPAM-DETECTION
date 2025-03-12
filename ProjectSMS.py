import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

# 1 and 2:
def transformation(texts):
    texts = texts.lower().split()

    alpha_num = []

    for words in texts:
        if words.isalnum():
            alpha_num.append(words)

    filtered = []

    for items in alpha_num:
        if items not in stopwords.words('english') and items not in string.punctuation:
            filtered.append(items)

    dancing_danc = []

    for items in filtered:
        dancing_danc.append(ps.stem(items))

    return " ".join(dancing_danc)


tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")
sms_input = st.text_input("Enter the message")

#1. preprocess
transformed_sms = transformation(sms_input)


#2. Vectorise
vector_input = tfidf.transform([transformed_sms])

#3. Predict                                    
result = model.predict(vector_input)[0]
#4. Display


if result == 1:
    st.header('SPAM!!')
elif result == 0:
    st.header("NOT SPAM!")
