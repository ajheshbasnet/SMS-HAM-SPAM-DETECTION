import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()

# Preprocessing function
def transformation(texts):
    texts = texts.lower().split()
    alpha_num = [word for word in texts if word.isalnum()]
    filtered = [word for word in alpha_num if word not in stopwords.words('english') and word not in string.punctuation]
    stemmed = [ps.stem(word) for word in filtered]
    return " ".join(stemmed)

# Load vectorizer and model
tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('Finalmodel.pkl', 'rb'))

# Custom CSS for dark background and visible fonts
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    h1 {
        color: #F39C12;
        text-align: center;
    }
    .stTextArea textarea {
        background-color: #2c2f33;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.write("Paste or type a message below and click the button to check if it's spam.")

# Input
sms_input = st.text_area("Message", height=200, placeholder="Type or paste your message here...")

if st.button("🚀 Check Message") and sms_input:
    # Preprocess
    transformed_sms = transformation(sms_input)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]
    confidence = max(model.predict_proba(vector_input)[0]) * 100  # Optional confidence score

    # Display result
    if result == 1:
        st.error(f"❌ SPAM MESSAGE DETECTED!!")
    else:
        st.success(f"✅ This message is NOT SPAM!!")