from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

emails = [
    "Win a free lottery now",      # Spam
    "Exclusive offer just for you",# Spam
    "Claim your free prize today", # Spam
    "Meeting at 10 AM tomorrow",   # Not Spam
    "Project deadline is next week", # Not Spam
    "Let's catch up for coffee",   # Not Spam
]

labels = [1, 1, 1, 0, 0, 0]

#Converts text data into numerical features using CountVectorizer (Bag of Words).
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.3, random_state=42)

#Uses Multinomial Naïve Bayes to learn from the dataset.
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

new_email = ['Meet with me later']
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)

print("Prediction", "Spam" if prediction[0] == 1 else "Not Spam")