import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from src.preprocess import preprocess_data

def train_model(data_path):
    
    df = pd.read_csv(data_path)

    X = df['Message']
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tfidf, X_test_tfidf, vectorizer = preprocess_data(X_train, X_test)


    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)


    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    joblib.dump(model, 'models/spam_detection_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

if __name__ == "__main__":
    train_model('data/spam.csv')
