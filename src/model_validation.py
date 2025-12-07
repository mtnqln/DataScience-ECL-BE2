from prepare_data import embeddings_creux, embeddings_dense, embedding_query_dense
from handle_data import load_corpus, load_queries, load_qrels

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np


def validate_model(queries, corpus, valid, model_type='dense'):
    if model_type == 'dense':
        embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    elif model_type == 'creux':
        embeddings, model = embeddings_creux(corpus)
    else:
        raise ValueError("model_type not valid")

    print("Embeddings loaded.")
    corpus_ids = list(corpus.keys())
    all_true_labels = []
    all_pred_labels = []
    all_pred_continuous_labels = []

    for query_id in valid.keys():
        # Création du vecteur de la requête
        query = queries[query_id]
        query_text = query['text']
        print(f"Processing query ID: {query_text}")
        
        if model_type == 'dense':
            # query_vector = model.encode([query_text])
            query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        elif model_type == 'creux':
            query_vector = model.transform([query_text])

        # Création des vecteurs des documents candidats
        query_candidates_id = valid[query_id].keys()
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings[candidate_index]

        # Score de similarité entre la requête et les documents candidats
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        # On choisit les 5 meilleurs scores
        scores_sorted = np.sort(scores)[::-1]
        best_scores = scores_sorted[:5]

        all_true_labels.extend([valid[query_id][candidate_id] for candidate_id in query_candidates_id])
        all_pred_continuous_labels.extend(scores)

        all_pred_labels.extend([1 if score in best_scores else 0 for score in scores])
        # all_pred_labels.extend([1 if score >= 0.2799 else 0 for score in scores])

    # Calcul des métriques de performance de prédiction
    precision = precision_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    recall = recall_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    f1 = f1_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    auc_score = roc_auc_score(np.array(all_true_labels), np.array(all_pred_continuous_labels))

    print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nAUC Score: {auc_score}")


if __name__ == "__main__":
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    qrels_valid = load_qrels("data/valid.tsv")

    validate_model(queries, corpus, qrels_valid, model_type='dense')