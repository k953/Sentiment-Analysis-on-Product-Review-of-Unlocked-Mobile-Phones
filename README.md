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



📱 Amazon Customer Sentiment Analysis
✅ Machine Learning + Word2Vec + LSTM (Complete NLP Pipeline)

This project analyzes Amazon mobile product reviews and predicts whether the customer sentiment is Positive or Negative using NLP techniques including CountVectorizer, Naive Bayes, Word2Vec, and LSTM Deep Learning models.

✅ 1. Project Overview

Customer reviews contain valuable signals about product quality, brand perception, and user satisfaction.
This project builds an end-to-end sentiment analysis system with:

✅ Extensive Data Exploration
✅ Data Cleaning & Preprocessing
✅ Machine Learning (Naive Bayes)
✅ Word Embeddings (Word2Vec)
✅ Deep Learning (LSTM, LSTM + Word2Vec)
✅ WordCloud visualization

The final LSTM + Word2Vec model achieves 94.40% accuracy.

✅ 2. Dataset
Amazon Unlocked Mobile Reviews Dataset (from Kaggle)

Total Reviews: 413,840
Total Brands: 385
Unique Products: 4,410
Columns: Product Name, Brand, Price, Rating, Reviews, Votes

✅ Sentiment Distribution
Rating	Meaning	Percentage
1–2	Negative	23.45%
3	Neutral	7.68%
4–5	Positive	68.86%

Neutral reviews were removed (Rating = 3).

✅ 3. Data Exploration (EDA)
✅ Summary Statistics

Average Price: $226.86
Average Rating: 3.81
Review Votes Range: 0 to 645

✅ Visualizations

Rating Distribution
Top 20 Most Reviewed Brands
Top 50 Most Reviewed Products
Review Length Distribution

✅ 4. Data Preparation
✅ Optional Sampling

df = df.sample(frac=0.1, random_state=0)

✅ Remove Missing & Neutral Reviews

df.dropna(inplace=True)
df = df[df['Rating'] != 3]

✅ Encode Sentiment

df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)

✅ Train-Test Split

90% Train, 10% Test

✅ 5. Text Preprocessing

Cleaning included:
HTML removal
Special character removal
Lowercasing
Stopwords removal
Stemming (Snowball Stemmer)

Function used:
cleanText(raw_text)

✅ 6. Machine Learning Model
✅ CountVectorizer + Multinomial Naive Bayes

Accuracy: 91.84%
AUC Score: 0.879

A strong baseline model.

✅ 7. Word2Vec Embedding

Word2Vec trained with:
Vector Size = 300
Window = 10
Min Word Count = 10

Vocabulary Size = 4016 words
Reviews converted to 300-dimensional vectors.

✅ 8. Deep Learning Models
⭐ LSTM with Keras Embedding

Accuracy: 94.14%

⭐ LSTM with Word2Vec Embedding (Best Model)

Accuracy: 94.40%

✅ 9. WordCloud Visualization

Brand-wise word clouds created using:
create_word_cloud("Apple", sentiment=1)

✅ 10. Final Model Comparison
Model	Technique	Accuracy
Naive Bayes	CountVectorizer	91.84%
LSTM	Keras Embedding	94.14%
⭐ LSTM + Word2Vec	Custom Embedding	94.40%
✅ 11. Technologies Used

Python
Pandas, NumPy
Scikit-Learn
NLTK
BeautifulSoup
Gensim Word2Vec
TensorFlow / Keras
Matplotlib, Seaborn
WordCloud

✅ 12. Future Enhancements

Transformer models (BERT, RoBERTa)
Real-time sentiment dashboard
Multi-class rating prediction
Brand-wise recommendations
