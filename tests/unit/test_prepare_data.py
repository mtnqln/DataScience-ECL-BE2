from src.handle_data import *
from src.prepare_data import *

def test_prepare_for_vectorizer():
    file_path = "data/mock_prepare_data.jsonl"
    corpus = load_corpus(file_path)
    titles = prepare_for_vectorizer(corpus)
    
    expected_result = [
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0]
    ]
    print(titles)
    #assert tab.toarray().tolist() == expected_result
def test_vectorize_data():
    file_path = "data/mock_prepare_data.jsonl"
    corpus = load_corpus(file_path)
    corpus_text_prepared = prepare_for_vectorizer(corpus)

    matrix, vectorizer  = vectorize_data(corpus_text_prepared)

    print(vectorizer)

# def test_vectorize_with_sentence_transformer():
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     matrix = model.encode(corpus_text)
#     return matrix, model