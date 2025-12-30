from .prepare_data import embeddings_creux, embeddings_dense, embedding_query_dense
from .handle_data import load_corpus, load_queries, load_qrels, load_sample_submission

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import pandas as pd

from .model_graph import build_graph, improve_embedding


def validate_model(queries, corpus, valid, model_type='dense'):
    """
    Validate the base model (sparse or dense) without graph enhancement.
    
    Args:
        queries: Dictionary of query documents
        corpus: Dictionary of corpus documents
        valid: Validation set with relevance judgments
        model_type: 'dense' or 'creux' for embedding type
        
    Returns:
        Dictionary with precision, recall, f1, and auc scores
    """
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
        print(f"Processing query ID: {query_id[:50]}...")
        
        if model_type == 'dense':
            query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        elif model_type == 'creux':
            query_vector = model.transform([query_text])

        # Création des vecteurs des documents candidats
        query_candidates_id = list(valid[query_id].keys())
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

    # Calcul des métriques de performance de prédiction
    precision = precision_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    recall = recall_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    f1 = f1_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    auc_score = roc_auc_score(np.array(all_true_labels), np.array(all_pred_continuous_labels))

    print(f"\n{'='*50}")
    print(f"BASE MODEL RESULTS ({model_type})")
    print(f"{'='*50}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"{'='*50}\n")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }

def validate_graph_model(queries, corpus, valid, model_type='dense'):
    """
    Validate the model using graph-enhanced embeddings.
    
    This function improves document embeddings by incorporating citation graph structure,
    then evaluates the search engine performance on the validation set.
    
    Args:
        queries: Dictionary of query documents
        corpus: Dictionary of corpus documents
        valid: Validation set with relevance judgments
        model_type: 'dense' or 'creux' for embedding type
    """
    if model_type == 'dense':
        embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    elif model_type == 'creux':
        embeddings, model = embeddings_creux(corpus)
    else:
        raise ValueError("model_type not valid")
    
    print("Building citation graph...")
    g = build_graph(corpus=corpus)
    
    # CRITICAL FIX: Create correct mapping from document ID to embedding index
    corpus_ids = list(corpus.keys())
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    
    print("Improving embeddings with graph structure...")
    embeddings = improve_embedding(corpus_embeddings=embeddings, G=g, id_to_index=id_to_index)

    print("Graph-enhanced embeddings loaded.")
    all_true_labels = []
    all_pred_labels = []
    all_pred_continuous_labels = []

    for query_id in valid.keys():
        # Création du vecteur de la requête
        query = queries[query_id]
        query_text = query['text']
        print(f"Processing query ID: {query_id[:50]}...")
        
        if model_type == 'dense':
            query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        elif model_type == 'creux':
            query_vector = model.transform([query_text])
        
        # If query is in the graph, improve its embedding too
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            # Use the graph-improved embedding for the query
            query_vector = embeddings[query_idx:query_idx+1]
            print(f"  Using graph-enhanced embedding for query {query_id[:50]}")

        # Création des vecteurs des documents candidats
        query_candidates_id = list(valid[query_id].keys())
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

    # Calcul des métriques de performance de prédiction
    precision = precision_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    recall = recall_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    f1 = f1_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    auc_score = roc_auc_score(np.array(all_true_labels), np.array(all_pred_continuous_labels))

    print(f"\n{'='*50}")
    print(f"GRAPH-ENHANCED MODEL RESULTS ({model_type})")
    print(f"{'='*50}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"{'='*50}\n")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }


def sample_prediction(queries, corpus, valid, model_type='dense'):
    if model_type == 'dense':
        embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    elif model_type == 'creux':
        embeddings, model = embeddings_creux(corpus)
    else:
        raise ValueError("model_type not valid")

    print(valid)

    print("Embeddings loaded.")
    corpus_ids = list(corpus.keys())
    all_true_labels = []
    all_pred_labels = []
    all_pred_continuous_labels = []

    for query_id in valid['query-id'].unique():
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
        query_candidates_id = valid[valid['query-id'] == query_id]['corpus-id'].tolist()
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings[candidate_index]

        # Score de similarité entre la requête et les documents candidats
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        # On choisit les 5 meilleurs scores
        scores_sorted = np.sort(scores)[::-1]
        best_scores = scores_sorted[:5]

        for i, candidate_id in enumerate(query_candidates_id): 
            valid.loc[(valid['query-id'] == query_id) & (valid['corpus-id'] == candidate_id), 'score'] = 1 if scores[i] in best_scores else 0
            # valid.loc[(valid['query-id'] == query_id) & (valid['corpus-id'] == candidate_id), 'score'] = scores[i]
    
    valid.to_csv('data/sample_submission_predicted.csv', index=False)


def sample_prediction_graph(queries, corpus, valid, model_type='dense'):
    """
    Generate predictions using graph-enhanced embeddings for sample submission.
    
    Args:
        queries: Dictionary of query documents
        corpus: Dictionary of corpus documents
        valid: DataFrame with query-id and corpus-id columns
        model_type: 'dense' or 'creux' for embedding type
        
    Returns:
        DataFrame with predictions
    """
    if model_type == 'dense':
        embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    elif model_type == 'creux':
        embeddings, model = embeddings_creux(corpus)
    else:
        raise ValueError("model_type not valid")
    
    print("Building citation graph for enhanced predictions...")
    g = build_graph(corpus=corpus)
    
    # Create correct mapping from document ID to embedding index
    corpus_ids = list(corpus.keys())
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    
    print("Improving embeddings with graph structure...")
    embeddings = improve_embedding(corpus_embeddings=embeddings, G=g, id_to_index=id_to_index)

    print("Graph-enhanced embeddings loaded.")
    print(f"Processing {len(valid['query-id'].unique())} queries...")

    for query_id in valid['query-id'].unique():
        # Création du vecteur de la requête
        query = queries[query_id]
        query_text = query['text']
        print(f"Processing query ID: {query_id[:50]}...")
        
        if model_type == 'dense':
            query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        elif model_type == 'creux':
            query_vector = model.transform([query_text])
        
        # If query is in the graph, improve its embedding too
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings[query_idx:query_idx+1]
            print(f"  Using graph-enhanced embedding for query")

        # Création des vecteurs des documents candidats
        query_candidates_id = valid[valid['query-id'] == query_id]['corpus-id'].tolist()
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings[candidate_index]

        # Score de similarité entre la requête et les documents candidats
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        # On choisit les 5 meilleurs scores
        scores_sorted = np.sort(scores)[::-1]
        best_scores = scores_sorted[:5]

        for i, candidate_id in enumerate(query_candidates_id): 
            valid.loc[(valid['query-id'] == query_id) & (valid['corpus-id'] == candidate_id), 'score'] = 1 if scores[i] in best_scores else 0
    
    output_file = 'data/sample_submission_predicted.csv'
    valid.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")
    
    return valid



if __name__ == "__main__":
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    qrels_valid = load_qrels("data/valid.tsv")
    # sample_prediction_data = pd.read_csv("data/sample_submission.csv")

    validate_graph_model(queries, corpus, qrels_valid, model_type='dense')
    # sample_prediction(queries, corpus, sample_prediction_data, model_type='dense')