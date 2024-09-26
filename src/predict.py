import joblib

def predict_spam(message):
    model = joblib.load('models/spam_detection_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    message_tfidf = vectorizer.transform([message])

    prediction = model.predict(message_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == "__main__":
    message = input("Enter a message to check: ")
    result = predict_spam(message)
    print(f"Prediction: {result}")
