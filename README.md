# Sentiment-Analysis-on-Product-Review-of-Unlocked-Mobile-Phones


# 🧠 Sentiment Analysis on Amazon Product Reviews using Word2Vec and LSTM

An end-to-end **NLP project** that predicts customer sentiment (positive or negative) from Amazon product reviews using traditional ML and Deep Learning approaches.  
The pipeline covers data cleaning, feature engineering, Word2Vec embedding, and LSTM modeling — visualized with Word Clouds for better insights.

---

## 📊 Project Overview

This project demonstrates the complete process of **sentiment classification** on Amazon reviews using:
1. **CountVectorizer + Multinomial Naive Bayes** (Baseline)
2. **Simple LSTM (learned embeddings)**
3. **LSTM with Word2Vec embeddings**

---

## 📁 Table of Contents
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Code Overview](#code-overview)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [License](#license)

---

## 🧾 Dataset

Each record contains:
| Column | Description |
|--------|--------------|
| Product Name | Product title |
| Brand Name | Product brand |
| Price | Price of the product |
| Rating | User rating (1–5 stars) |
| Reviews | Text of the review |
| Review Votes | Number of votes |
| Sentiment | Target label (0 = Negative, 1 = Positive) |

Neutral reviews (rating = 3) were removed.

---

## ⚙️ Pipeline

### **1. Data Preparation**
```python
df = df[df['Rating'] != 3]
df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['Sentiment'], test_size=0.1, random_state=0)



2. Text Cleaning

Remove HTML tags

Keep only alphabets

Convert to lowercase

Remove stopwords

Apply stemming (optional)

text = BeautifulSoup(raw_text, 'lxml').get_text()
letters_only = re.sub("[^a-zA-Z]", " ", text)

3. Baseline Model (Naive Bayes)
countVect = CountVectorizer()
X_train_vect = countVect.fit_transform(X_train_cleaned)
mnb = MultinomialNB()
mnb.fit(X_train_vect, y_train)

4. Word2Vec Embedding
w2v = Word2Vec(sentences, size=300, window=10, min_count=10, sample=1e-3)
w2v.save("w2v_300features_10minwordcounts_10context")

5. Deep Learning (LSTM)
model = Sequential([
    Embedding(top_words, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(2, activation='softmax')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

🧩 Results
Model	Feature Type	Accuracy	AUC	Notes
Multinomial NB	CountVectorizer	91.84%	0.879	Strong baseline
Simple LSTM	Learned Embedding	94.14%	—	Captures sequence patterns
LSTM + Word2Vec	Pretrained Embedding	94.40%	—	Adds semantic context
🌈 Visualizations
Distribution of Review Length
<img src="images/review_length.png" width="600"/>
Word Clouds

Generate Word Clouds by brand and sentiment:

create_word_cloud('Samsung', sentiment=1)  # Positive
create_word_cloud('Samsung', sentiment=0)  # Negative

Positive Reviews	Negative Reviews
<img src="images/positive_wc.png" width="300"/>	<img src="images/negative_wc.png" width="300"/>
💻 Installation
1. Clone the Repository
git clone https://github.com/yourusername/sentiment-analysis-word2vec-lstm.git
cd sentiment-analysis-word2vec-lstm

2. Install Dependencies
pip install -r requirements.txt

3. Run the Notebook

Open in Jupyter or Google Colab:

jupyter notebook notebooks/Sentiment_Analysis.ipynb

🧠 Key Insights

Removing neutral ratings improves classification clarity.

Word2Vec embeddings provide slight but consistent improvements.

LSTM outperforms traditional ML models by capturing word order and contextual meaning.

🔮 Future Work

Integrate GloVe or FastText embeddings

Add Bidirectional LSTM or Attention Mechanism

Compare with BERT / DistilBERT fine-tuning

Deploy as a web app using Streamlit



