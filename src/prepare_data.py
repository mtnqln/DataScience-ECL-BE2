from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def vectorize_text_data(corpus):
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return  vectorizer, matrix
