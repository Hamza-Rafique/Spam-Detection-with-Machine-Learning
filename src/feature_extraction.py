from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(corpus):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(corpus)
