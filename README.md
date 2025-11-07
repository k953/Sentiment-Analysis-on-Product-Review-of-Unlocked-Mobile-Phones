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




Amazon Customer Sentiment Analysis

Machine Learning, Word2Vec, and LSTM Based NLP Project

1. Project Overview

This project performs sentiment analysis on Amazon mobile product reviews.
The objective is to classify each review as Positive or Negative using multiple NLP approaches:

Data Exploration and Visualization

Text Preprocessing and Cleaning

Machine Learning Model (Naive Bayes)

Word2Vec Embeddings

LSTM Deep Learning Models

WordCloud Visualization

The final LSTM using Word2Vec embeddings achieves 94.40 percent accuracy.

2. Dataset

Dataset used: Amazon Unlocked Mobile Reviews (from Kaggle)

Key statistics
Total Reviews: 413840
Total Brands: 385
Unique Products: 4410
Neutral Reviews (rating = 3) were removed
Sentiment Labels
Rating 4 and 5 = Positive (1)
Rating 1 and 2 = Negative (0)

3. Data Exploration

Summary statistics of Price, Rating, and Review Votes were generated.
Rating distribution, top brands, top products, and review length distribution were visualized.

Insights
More than 68 percent reviews are positive
Around 23 percent reviews are negative
Only 7 percent reviews are neutral

4. Data Preparation

Steps

Optional 10 percent sampling

Remove missing values

Remove neutral reviews (rating = 3)

Encode sentiment (positive = 1, negative = 0)

Train test split using 90 percent train and 10 percent test

5. Text Preprocessing

A custom text cleaning function was used to perform
HTML removal
Special character removal
Lowercasing
Stopwords removal
Stemming using Snowball Stemmer

Both training and testing reviews were cleaned using this function.

6. Machine Learning Model

CountVectorizer with Multinomial Naive Bayes

Steps

Convert text into bag of words using CountVectorizer

Train MultinomialNB classifier

Evaluate on test set

Results
Accuracy: 91.84 percent
AUC Score: 0.879
F1 scores indicate strong performance on positive sentiment

7. Word2Vec Embedding

Reviews were split into sentences using NLTK tokenizer

Word2Vec model trained with
Vector size = 300
Window = 10
Minimum word count = 10

Vocabulary size generated = 4016 words

Average Word2Vec embedding was computed for each review
Resulting feature dimension = 300

8. Deep Learning Models
Model 1: LSTM with Keras Embedding

Tokenizer and padded sequences were created
LSTM architecture used
Embedding layer
LSTM layer
Dense output layer with softmax

Result
Accuracy: 94.14 percent

Model 2: LSTM with Word2Vec Embedding

Steps

Load trained Word2Vec matrix

Use it as weights for embedding layer

Build LSTM model

Train and evaluate

Result
Accuracy: 94.40 percent (best performing model)

9. WordCloud Visualization

Brand wise sentiment word clouds were created.
Example: Apple positive sentiment word cloud.

10. Final Model Comparison

Naive Bayes (CountVectorizer) = 91.84 percent
LSTM (Keras embedding) = 94.14 percent
LSTM with Word2Vec embedding = 94.40 percent

11. Key Takeaways

Deep learning models outperform traditional ML models
Word2Vec improves semantic understanding of text
Amazon reviews dataset is highly imbalanced toward positive reviews
LSTM is effective for long text sentiment understanding

12. Technologies Used

Python
Pandas
NumPy
NLTK
BeautifulSoup
Scikit Learn
Gensim Word2Vec
TensorFlow Keras
Matplotlib and Seaborn
WordCloud library

13. Future Improvements

Use transformer models like BERT or RoBERTa
Multi class sentiment prediction (ratings 1 to 5)
Real time sentiment dashboard
Brand wise recommendation engine
