from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the model and vectorizer
model = joblib.load("models/spam_detection_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/predict/")
async def predict(message: Message):
  
    vectorized_message = vectorizer.transform([message.text])
    prediction = model.predict(vectorized_message)
    return {"spam": int(prediction[0])}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Spam Detection API!"}
