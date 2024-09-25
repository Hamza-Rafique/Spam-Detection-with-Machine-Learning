from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test
