# 📱 Amazon Customer Sentiment Analysis  
### ✅ Machine Learning + Word2Vec + LSTM (Complete NLP Pipeline)

This project analyzes **Amazon mobile product reviews** and predicts whether the customer sentiment is **Positive** or **Negative** using multiple NLP techniques including **CountVectorizer**, **Naive Bayes**, **Word2Vec**, and **LSTM Deep Learning models**.

---

# ✅ 1. Project Overview

Customer reviews contain valuable signals about **product quality, brand perception, and user satisfaction**.  
This project builds an **end-to-end sentiment analysis system** with:

- ✅ Extensive Data Exploration  
- ✅ Data Cleaning & Preprocessing  
- ✅ Machine Learning (Naive Bayes)  
- ✅ Word Embeddings (Word2Vec)  
- ✅ Deep Learning (LSTM, LSTM + Word2Vec)  
- ✅ WordCloud visualization  

The final LSTM + Word2Vec model achieves **94.40% accuracy**.

---

# ✅ 2. Dataset

- **Amazon Unlocked Mobile Reviews Dataset** (from Kaggle)  
- **Total Reviews:** 413,840  
- **Total Brands:** 385  
- **Unique Products:** 4,410  
- **Columns:** Product Name, Brand, Price, Rating, Reviews, Votes

### ✅ Sentiment Distribution
| Rating | Meaning | % |
|--------|---------|-----|
| 1–2 | Negative | 23.45% |
| 3 | Neutral | 7.68% |
| 4–5 | Positive | 68.86% |

Neutral reviews (Rating = 3) were removed.

---

# ✅ 3. Data Exploration (EDA)

### ✅ Summary Statistics
- Avg Price: **$226.86**  
- Avg Rating: **3.81**  
- Review Votes range: **0 to 645**

### ✅ Visualizations
- Distribution of Rating  
- Top 20 Most Reviewed Brands  
- Top 50 Most Reviewed Products  
- Review Length Distribution  

These plots help understand data imbalance and review patterns.

---

# ✅ 4. Data Preparation

### ✅ Sampling (optional)
```python
df = df.sample(frac=0.1, random_state=0)


✅ Drop Missing Values
df.dropna(inplace=True)

✅ Remove Neutral Reviews
df = df[df['Rating'] != 3]

✅ Sentiment Encoding
df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)

✅ Train/Test Split
train_test_split(df['Reviews'], df['Sentiment'], test_size=0.1)

✅ 5. Text Preprocessing

A custom cleaning function performs:

HTML removal

Special character removal

Lowercasing

Stopword removal

Snowball stemming




📱 Amazon Customer Sentiment Analysis
✅ Machine Learning + Word2Vec + LSTM (Complete NLP Pipeline)

This project analyzes Amazon mobile product reviews and predicts whether the customer sentiment is Positive or Negative using multiple NLP techniques including CountVectorizer, Naive Bayes, Word2Vec, and LSTM Deep Learning models.

✅ 1. Project Overview

Customer reviews provide valuable signals about product quality, user experience, brand satisfaction, and overall trust.
This project builds an end-to-end sentiment analysis pipeline that includes:

✅ Extensive Data Exploration

✅ Data Cleaning & Text Preprocessing

✅ Machine Learning using Naive Bayes

✅ Word Embeddings using Word2Vec

✅ Deep Learning (LSTM & LSTM with Word2Vec)

✅ WordCloud Visualizations

The final LSTM + Word2Vec model achieves 94.40% accuracy, making it the best-performing approach.

✅ 2. Dataset

Dataset used: Amazon Unlocked Mobile Reviews (Kaggle)

📊 Dataset Stats

Total Reviews: 413,840

Total Brands: 385

Unique Products: 4,410

Columns: Product Name, Brand, Price, Rating, Reviews, Votes

✅ Sentiment Mapping
Rating	Sentiment	Meaning
1–2	Negative (0)	Bad product experience
3	Neutral	Removed from dataset
4–5	Positive (1)	Good product experience

Neutral reviews (Rating = 3) were removed for clear binary classification.

✅ 3. Data Exploration (EDA)
📌 Summary Statistics

Average Price: $226.86

Average Rating: 3.81

Review Votes range: 0 to 645

📈 EDA Visualizations

Distribution of ratings

Top 20 brands with most reviews

Top 50 most reviewed products

Distribution of review lengths

These help understand imbalance and user reviewing behavior.

✅ 4. Data Preparation
✅ Sampling (Optional)
df = df.sample(frac=0.1, random_state=0)

✅ Removing Missing & Neutral Data
df.dropna(inplace=True)
df = df[df['Rating'] != 3]

✅ Encoding Sentiment
df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)

✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['Reviews'], df['Sentiment'], test_size=0.1, random_state=0
)

✅ 5. Text Preprocessing

A custom cleaning function was used for:

Removing HTML

Removing non-alphabet characters

Lowercasing

Removing stopwords

Stemming using Snowball Stemmer

✅ Text Cleaning Example
cleanText(raw_text)

✅ 6. Machine Learning Model (Baseline)
🎯 CountVectorizer + Multinomial Naive Bayes
✅ Vectorization
countVect = CountVectorizer()
X_train_vec = countVect.fit_transform(X_train_cleaned)

✅ Model Training
mnb = MultinomialNB()
mnb.fit(X_train_vec, y_train)

✅ Evaluation Results

Accuracy: 91.84%

AUC Score: 0.879

Well-performing baseline

✅ 7. Word2Vec Embeddings
✅ Sentence Tokenization
sentences = tokenizer.tokenize(review)

✅ Training Word2Vec
w2v = Word2Vec(size=300, window=10, min_count=10)

✅ Vocabulary Size

4016 words

✅ Convert Each Review → 300-Dimensional Vector
makeFeatureVec(review, model, 300)

✅ 8. Deep Learning Models
⭐ Model 1: LSTM with Keras Embedding
✅ Tokenization + Padding
tokenizer = Tokenizer(num_words=20000)
pad_sequences(sequences, maxlen=100)

✅ LSTM Architecture
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128))
model.add(Dense(2, activation="softmax"))

✅ Accuracy

✅ 94.14%

⭐ Model 2: LSTM with Word2Vec Embedding (Best Model)
✅ Embedding Loaded from Word2Vec
Embedding(vocab_size, 300, weights=[embedding_matrix])

✅ Accuracy

✅ 94.40%

This is the best-performing model.

✅ 9. WordCloud Visualization

Brand-wise & sentiment-wise keyword clouds created using:

create_word_cloud(brand='Apple', sentiment=1)


Example insights:
Apple positive reviews include: great, fast, quality, love.

✅ 10. Final Performance Summary
Model	Technique	Accuracy
Naive Bayes	CountVectorizer	91.84%
LSTM	Keras Embedding	94.14%
LSTM + Word2Vec	Custom Embedding	⭐ 94.40%
✅ 11. Technologies Used

Python

Pandas, NumPy

BeautifulSoup

NLTK

Scikit-Learn

Gensim (Word2Vec)

TensorFlow / Keras

Matplotlib

Seaborn

WordCloud

✅ 12. Future Enhancements

Add BERT / RoBERTa transformer models

Build live sentiment dashboard

Product recommendation based on sentiment

Multi-class rating prediction (1–5 stars)
