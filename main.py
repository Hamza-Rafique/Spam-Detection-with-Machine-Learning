import pandas as pd
from src.data_preprocessing import preprocess_text
from src.feature_extraction import extract_features
from src.train_model import train_model
from src.evaluate_model import evaluate_model

# Load dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df['message_cleaned'] = df['message'].apply(preprocess_text)

# Feature extraction
X = extract_features(df['message_cleaned'])
y = df['label']

# Train the model
model, X_test, y_test = train_model(X, y)

# Evaluate model
report = evaluate_model(model, X_test, y_test)
print(report)
