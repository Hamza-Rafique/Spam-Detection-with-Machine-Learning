import nltk
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char not in string.punctuation])
    tokens = text.split()
    text = [word for word not in stopwords.words('english')]
    return ' '.join(text)
