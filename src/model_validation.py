from model import search_engine
from prepare_data import vectorize_data, prepare_for_vectorizer
from handle_data import load_corpus, load_queries, load_qrels

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def validate_model(queries, corpus, valid):
    corpus_text = prepare_for_vectorizer(corpus)
    matrix, vectorizer = vectorize_data(corpus_text)

    all_true_labels = []
    all_pred_labels = []
    all_pred_continuous_labels = []

    for query_id in valid.keys():
        # Création du vecteur de la requête
        query = queries[query_id]
        query_text = query['text']
        query_vector = vectorizer.transform([query_text])

        # Création des vecteurs des documents candidats
        query_candidates_id = valid[query_id].keys()
        query_candidates = {key: corpus[key] for key in query_candidates_id}
        query_candidates_text = prepare_for_vectorizer(query_candidates)
        query_candidate_vectors = vectorizer.transform(query_candidates_text)

        # Score de similarité entre la requête et les documents candidats
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()

        results_df = pd.DataFrame({
            'id': query_candidates_id,
            'score': scores,
            'title': [corpus[id]['title'] for id in query_candidates_id]
        })

        # On choisit les 5 meilleurs scores 
        top_5_indices = results_df['score'].nlargest(5).index
        results_df['score_binaire'] = 0
        results_df.loc[top_5_indices, 'score_binaire'] = 1

        # On attribue les labels vrais et prédits
        for candidate_id in query_candidates_id:
            # # Titre du candidat
            # print(corpus[candidate_id]['title'])
            # # Vrai score
            # print(valid[query_id][candidate_id])
            # # Score binaire prédit
            # print(results_df[results_df['id'] == candidate_id]['score_binaire'].values[0])
            # # Score continu prédit
            # print(results_df[results_df['id'] == candidate_id]['score'].values[0])

            all_true_labels.append(valid[query_id][candidate_id])
            all_pred_labels.append(results_df[results_df['id'] == candidate_id]['score_binaire'].values[0])
            all_pred_continuous_labels.append(results_df[results_df['id'] == candidate_id]['score'].values[0])

    # Calcul des métriques de performance de prédiction
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)
    y_pred_continuous = np.array(all_pred_continuous_labels)

    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc_score = roc_auc_score(y_true, y_pred_continuous)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC Score: {auc_score}")


if __name__ == "__main__":
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    qrels_valid = load_qrels("data/valid.tsv")

    validate_model(queries, corpus, qrels_valid)