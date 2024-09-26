from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(train_texts, test_texts):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    return X_train_tfidf, X_test_tfidf, vectorizer
