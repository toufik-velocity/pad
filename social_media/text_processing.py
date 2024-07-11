# text_processing.py

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# tokenizer
def tokenize_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens
