# 🧠 Sentiment Analysis on Amazon Product Reviews using Word2Vec and LSTM

An end-to-end **Natural Language Processing (NLP)** project that predicts whether a product review on Amazon is **positive or negative**.  
This project demonstrates the full machine learning and deep learning workflow — from data cleaning, feature extraction (Word2Vec), and LSTM-based sentiment classification, to visualization using Word Clouds.

---

## 📋 Project Overview

The goal of this project is to develop an intelligent system that can understand human sentiment from product reviews.  
We compare traditional ML and deep learning approaches:

| Model | Feature Type | Accuracy | AUC | Notes |
|--------|---------------|----------|------|--------|
| **Multinomial Naive Bayes** | CountVectorizer | 91.84% | 0.879 | Baseline model |
| **Simple LSTM** | Learned Embedding | 94.14% | – | Captures sequence context |
| **LSTM + Word2Vec** | Pretrained Word Embeddings | **94.40%** | – | Semantic improvement |

---

## ⚙️ Pipeline Summary

### **1. Data Preprocessing**
- Removed neutral reviews (rating = 3)
- Cleaned text (HTML removal, stopword removal, lowercasing)
- Created binary sentiment labels:
  - Ratings > 3 → Positive (1)
  - Ratings ≤ 2 → Negative (0)
  
```python
df = df[df['Rating'] != 3]
df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)



2. Baseline Model (Naive Bayes)

Used CountVectorizer for Bag-of-Words representation and trained a MultinomialNB model.

countVect = CountVectorizer()
X_train_vect = countVect.fit_transform(X_train_cleaned)
mnb = MultinomialNB()
mnb.fit(X_train_vect, y_train)



📊 Validation Accuracy: 91.84%
✅ A good classical ML benchmark before deep learning.

3. Word2Vec Embeddings

Trained Word2Vec on review sentences to capture semantic meaning.

w2v = Word2Vec(sentences, size=300, min_count=10, window=10, sample=1e-3, workers=4)
w2v.save("w2v_300features_10minwordcounts_10context")


🧩 Vocabulary Size: 4,016
🧠 Embedding Dimension: 300


4. Deep Learning Models
🔹 Simple LSTM
model1 = Sequential([
    Embedding(top_words, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(2, activation='softmax')
])
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


📈 Test Accuracy: 94.14%


🔹 LSTM with Word2Vec Embedding
embedding_layer = Embedding(w2v.wv.vectors.shape[0],
                            w2v.wv.vectors.shape[1],
                            weights=[w2v.wv.vectors])
model2 = Sequential([
    embedding_layer,
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(2, activation='softmax')
])
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


📈 Test Accuracy: 94.40%

✅ Word2Vec initialization slightly improved results by capturing semantic similarity between words.


📊 Evaluation Metrics
Metric	Score
Accuracy	94.40%
Precision (Positive)	0.93
Recall (Positive)	0.96
F1-Score (Positive)	0.95
AUC Score (Baseline)	0.879

Confusion Matrix Example:



☁️ Word Cloud Visualization

Generate brand-wise sentiment word clouds:

create_word_cloud('Samsung', sentiment=1)  # Positive Reviews
create_word_cloud('Samsung', sentiment=0)  # Negative Reviews

Positive Reviews	Negative Reviews

	
🧠 Key Insights

Neutral reviews removal improved clarity in classification.

LSTM models significantly outperformed Naive Bayes.

Word2Vec embeddings captured contextual meaning, improving accuracy slightly.

Visualization provided qualitative insights into customer sentiment patterns.

🧰 Libraries Used
numpy
pandas
matplotlib
seaborn
nltk
gensim
beautifulsoup4
wordcloud
keras
tensorflow
scikit-learn

📂 Project Structure
📁 sentiment-analysis-word2vec-lstm
├── data/
│   └── amazon_reviews.csv
├── models/
│   ├── w2v_300features_10minwordcounts_10context
│   ├── lstm_simple.h5
│   └── lstm_word2vec.h5
├── notebooks/
│   └── Sentiment_Analysis.ipynb
├── images/
│   ├── review_length.png
│   ├── positive_wc.png
│   └── negative_wc.png
├── utils/
│   └── text_preprocessing.py
├── requirements.txt
└── README.md

🧾 Installation
Step 1: Clone Repository
git clone https://github.com/yourusername/sentiment-analysis-word2vec-lstm.git
cd sentiment-analysis-word2vec-lstm

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Jupyter Notebook
jupyter notebook notebooks/Sentiment_Analysis.ipynb

🚀 Results Summary
Model	Accuracy	Remarks
Naive Bayes (CountVectorizer)	91.8%	Baseline
Simple LSTM (Embedding)	94.1%	Captures sequence
LSTM + Word2Vec	94.4%	Semantic improvement

✅ Final Model: LSTM + Word2Vec
📈 Performance: 94.40% accuracy
🌈 Visualization: Brand-wise WordClouds for positive & negative sentiments

💬 Conclusion

This project successfully demonstrates an end-to-end sentiment analysis workflow:

Preprocessing raw text data

Feature representation using Word2Vec

Deep learning model (LSTM) training

Model evaluation & visualization

Such pipelines can be extended to social media sentiment analysis, customer feedback analysis, and opinion mining.




