from prepare_data import vectorize_data, prepare_for_vectorizer, vectorize_with_sentence_transformer
from handle_data import load_corpus

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def search_engine(query_text: str, corpus):
    corpus_text = prepare_for_vectorizer(corpus)

    matrix, vectorizer = vectorize_data(corpus_text)
    query_vector = vectorizer.transform([query_text])

    # matrix, model = vectorize_with_sentence_transformer(corpus_text)
    # query_vector = model.encode([query_text])

    similarity_matrix = cosine_similarity(query_vector, matrix)

    doc_ids = list(corpus.keys())
    scores = similarity_matrix.flatten()


    # Créer un DataFrame pour combiner les IDs, scores et titres
    results_df = pd.DataFrame({
        'id': doc_ids,
        'score': scores,
        'title': [corpus[id]['title'] for id in doc_ids]
    })
    
    # Trier par score décroissant
    sorted_results = results_df.sort_values(by='score', ascending=False)

    print(f"Top 10 résultats pour la requête : '{query_text}'")
    
    # Afficher les top N résultats (sans le score 0 pour la requête elle-même)
    top_results = sorted_results[sorted_results['score'] > 0].head(10)
    print(top_results)

    return top_results


if __name__ == "__main__":
    corpus = load_corpus("data/corpus.jsonl")
    results = search_engine("cyber broccoli", corpus)
    # print(results)