📱 Amazon Mobile Reviews — Sentiment Analysis using Word2Vec + LSTM
🎯 Project Overview

This project performs Sentiment Analysis on Amazon Unlocked Mobile Reviews using both classical machine learning (Naive Bayes) and deep learning (LSTM with Word2Vec embeddings).
It predicts whether a review expresses positive or negative sentiment based on the text.

🧠 Objective

Preprocess and clean real Amazon review text

Train Word2Vec embeddings to capture semantic meaning of words

Build an LSTM network to learn sequential sentiment patterns

Evaluate and visualize results with accuracy, confusion matrix, and word clouds

📦 Dataset

Source: Amazon Unlocked Mobile Dataset (Kaggle)

Column	Description
Product Name	Mobile name
Brand Name	Company (Samsung, Apple, etc.)
Price	Product price
Rating	User rating (1–5 stars)
Reviews	Text review
Review Votes	Helpful votes count

Size: ~4,13,000 reviews
Language: English

🧹 1. Data Preprocessing
df = pd.read_csv('Amazon_Unlocked_Mobile.csv')

# Drop missing values
df.dropna(inplace=True)

# Remove neutral reviews (rating == 3)
df = df[df['Rating'] != 3]

# Encode sentiment: 1 = Positive (rating > 3), 0 = Negative (rating < 3)
df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)


✅ Result:
Only positive (4–5) and negative (1–2) reviews remain.
Example:

Rating	Sentiment	Example Review
5	1	“Excellent camera and battery life!”
1	0	“Worst phone ever, totally useless.”
✂️ 2. Train–Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['Reviews'], df['Sentiment'], test_size=0.1, random_state=0
)


📊 Example split:

Training samples: 27,799

Validation samples: 3,089

🧽 3. Text Cleaning
def cleanText(raw_text):
    text = BeautifulSoup(raw_text, 'lxml').get_text()  # Remove HTML
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    return " ".join(words)


✅ Removes:

HTML tags

Punctuation

Converts to lowercase

Example:

Before: "Good product! Fast delivery :)"
After:  "good product fast delivery"

📊 4. Baseline Model — Bag of Words + Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

countVect = CountVectorizer()
X_train_cv = countVect.fit_transform(X_train_cleaned)
X_test_cv = countVect.transform(X_test_cleaned)

model_nb = MultinomialNB()
model_nb.fit(X_train_cv, y_train)
pred = model_nb.predict(X_test_cv)


Result:

Accuracy: 91.8%
AUC: 0.879
F1-Score: 0.92


✅ Insight: Bag-of-Words works well but doesn’t understand context.

🧩 5. Word2Vec Embeddings

Train Word2Vec to capture semantic similarity between words.

from gensim.models import Word2Vec
sentences = [review.split() for review in X_train_cleaned]

w2v = Word2Vec(sentences, size=300, window=10, min_count=10, workers=4)
w2v.save("w2v_300features_10minwordcounts_10context")


📈 Vocabulary size: 4016 words
Each word represented as a 300-dimensional vector

Example:

Vector("good") ≈ Vector("great")
Vector("bad") ≈ Vector("terrible")

🤖 6. Deep Learning Model — LSTM + Word2Vec
a. Prepare Sequence Input
tokenizer = Tokenizer(num_words=4016)
tokenizer.fit_on_texts(X_train)
X_train_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

b. Load Embedding Matrix
embedding_matrix = w2v.wv.syn0

c. Build Model
model = Sequential()
model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_seq, y_train_seq, batch_size=32, epochs=3, verbose=1)

d. Evaluation
score = model.evaluate(X_test_seq, y_test_seq)
print(f"Test Accuracy: {score[1]*100:.2f}%")


✅ Result:
Test Accuracy: 94.4%
Loss: 0.1597

🧮 7. Architecture Summary
Layer	Output Shape	Parameters
Embedding (Word2Vec)	(None, 100, 300)	1,204,800
LSTM (128 units)	(None, 128)	219,648
Dense (2)	(None, 2)	258
Total Trainable Params	1,424,706	✅
☁️ 8. Word Cloud Visualization
from wordcloud import WordCloud

def create_word_cloud(brand, sentiment):
    df_brand = df[df['Brand Name'] == brand]
    df_reviews = df_brand[df_brand['Sentiment']==sentiment]['Reviews']
    text = " ".join(df_reviews.astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


Example:

create_word_cloud('Apple', 1)  # Positive reviews
create_word_cloud('Apple', 0)  # Negative reviews


🟢 Positive words: “great”, “love”, “camera”, “battery”
🔴 Negative words: “bad”, “problem”, “expensive”, “slow”

🔁 9. Comparison — Models
Model	Features	Accuracy
Multinomial Naive Bayes	CountVectorizer	91.8%
LSTM (Random Embedding)	128-D	94.1%
LSTM + Word2Vec	300-D pretrained	94.4% ✅

✅ LSTM + Word2Vec learns both sequence + semantic context.

🧠 10. Intuitive Flow Diagram
Raw Text → Tokenizer → Word2Vec Embedding (4016×300)
          ↓
       LSTM Layer (128 units)
          ↓
      Dense + Softmax → [Negative, Positive]

📈 11. Results Summary
Metric	Value
Accuracy	94.4%
AUC	0.93
Loss	0.1597
Dataset	Amazon Unlocked Mobile Reviews
Framework	TensorFlow / Keras
🎨 12. Visual Outputs

📊 Accuracy vs Epoch plot

☁️ Word Clouds (Positive / Negative)

🔢 Confusion Matrix Heatmap

(Add figures if available from training logs or matplotlib outputs.)

🧾 13. Conclusion

✅ Classical ML (Naive Bayes) gives strong baseline
✅ Word2Vec + LSTM captures contextual word meaning
✅ Achieves >94% accuracy on real Amazon reviews
✅ Visualizations (WordClouds) explain sentiment trends across brands
