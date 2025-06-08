# 📱 SMS Spam/Ham Detection

This is a machine learning web application that classifies text messages (SMS) as either **Spam** or **Ham (Not Spam)**.

The model is built using Natural Language Processing (NLP) techniques and a custom **stacked ensemble classifier**, then deployed on **Render**.

---

## 🚀 Live Demo

👉 Check out the live app here:  
**🔗 [https://sms-ham-spam-detection-1.onrender.com](https://sms-ham-spam-detection-1.onrender.com)**

---

## 🔍 Features

- Classifies SMS messages into **Spam** or **Ham**
- Uses a **stacking classifier** combining:
  - ✅ Extra Trees Classifier
  - ✅ Multinomial Naive Bayes
- Text preprocessing, TF-IDF transformation, and fast predictions
- Clean and simple web UI using Streamlit
- Cloud-hosted — works even when the local machine is off

---

## 🧠 Model Info

- **Text Preprocessing**:
  - Lowercasing
  - Removing punctuation
  - Removing stopwords (via NLTK)
  - Stemming using Porter Stemmer
- **Feature Extraction**:
  - TF-IDF Vectorization
- **Model Architecture**:
  - Stacked Ensemble using:
    - **Extra Trees Classifier**
    - **Multinomial Naive Bayes**
  - Combined using `sklearn`'s `StackingClassifier`

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend/ML**: Python, scikit-learn, NLTK
- **Deployment**: Render

---

## 📂 How to Run Locally

```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector
pip install -r requirements.txt
streamlit run app.py
