import string
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [word for word in words if word.isalpha()]
    return ' '.join(words)
