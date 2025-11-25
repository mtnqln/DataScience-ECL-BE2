from sklearn.feature_extraction.text import CountVectorizer


def vectorize_data(corpus):
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return matrix , vectorizer