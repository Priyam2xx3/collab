# 📧 SMS/Email Spam Classifier

A machine learning project that classifies SMS or email messages as **spam** or **not spam (ham)** using NLP techniques and supervised learning models.

---

## 🔍 Overview

This project uses a dataset of labeled messages and builds a classifier that can identify whether a message is spam. It demonstrates the full pipeline of preprocessing, feature extraction, model training, and evaluation.

---

## 📁 Dataset

- **Source**: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description**: Contains ~5,500 messages labeled as `spam` or `ham`.

---

## ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NLTK / spaCy (for NLP)
- Google colab / Streamlit (for UI, optional)

---

##  How It Works

1. **Text Preprocessing**:
   - Lowercasing
   - Removing punctuation, numbers, and stopwords
   - Tokenization and lemmatization

2. **Feature Extraction**:
   - Using TF-IDF Vectorizer to convert text to numerical features

3. **Model Training**:
   - Naive Bayes and Logistic Regression classifiers trained on labeled messages

4. **Evaluation**:
   - Measured using accuracy, precision, recall, and F1-score

5. **Prediction**:
   - The model can predict whether new messages are spam or ham

---

## 🚀 How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Google Colab or Streamlit app
google cplab spam_classifier.ipynb
# or
streamlit run app.py

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97.5% |
| Precision | 96.8% |
| Recall    | 95.2% |
| F1 Score  | 96.0% |

