from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return classification_report(y_test, predictions)
