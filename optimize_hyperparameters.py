"""
Script pour optimiser les hyperparamètres du modèle graphe.
Teste différentes configurations et génère les prédictions avec la meilleure.
"""

import pandas as pd
import numpy as np
from src.handle_data import load_corpus, load_queries, load_qrels
from src.prepare_data import embeddings_dense, embedding_query_dense
from src.model_graph import build_graph, improve_embedding, improve_embedding_advanced
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_config(queries, corpus, valid, embeddings_func, config_name):
    """Evaluate a specific configuration on validation set."""
    
    # Embeddings
    embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    
    # Graph
    g = build_graph(corpus=corpus)
    corpus_ids = list(corpus.keys())
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    
    # Apply embedding improvement
    embeddings = embeddings_func(embeddings, g, id_to_index)
    
    # Evaluate
    all_true_labels = []
    all_pred_labels = []
    all_pred_continuous_labels = []
    
    for query_id in valid.keys():
        query = queries[query_id]
        query_text = query['text']
        
        query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings[query_idx:query_idx+1]
        
        query_candidates_id = list(valid[query_id].keys())
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings[candidate_index]
        
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        scores_sorted = np.sort(scores)[::-1]
        best_scores = scores_sorted[:5]
        
        all_true_labels.extend([valid[query_id][candidate_id] for candidate_id in query_candidates_id])
        all_pred_continuous_labels.extend(scores)
        all_pred_labels.extend([1 if score in best_scores else 0 for score in scores])
    
    # Calculate metrics
    precision = precision_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    recall = recall_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    f1 = f1_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    auc_continuous = roc_auc_score(np.array(all_true_labels), np.array(all_pred_continuous_labels))
    auc_binary = roc_auc_score(np.array(all_true_labels), np.array(all_pred_labels))
    
    print(f"\nRésultats:")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  AUC (Prob):   {auc_continuous:.4f}")
    print(f"  AUC (Binaire): {auc_binary:.4f}")
    
    return {
        'config': config_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_prob': auc_continuous,
        'auc_kaggle': auc_binary
    }

def main():
    
    # Load data
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    qrels_valid = load_qrels("data/valid.tsv")
    
    # Test configurations
    configurations = [
        {
            'name': 'Baseline (alpha=0.3, beta=0.2)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.3, beta=0.2)
        },
        {
            'name': 'Higher weights (alpha=0.4, beta=0.3)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.4, beta=0.3)
        },
        {
            'name': 'More references (alpha=0.5, beta=0.2)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.5, beta=0.2)
        },
        {
            'name': 'More citations (alpha=0.3, beta=0.4)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.3, beta=0.4)
        },
        {
            'name': 'Advanced (alpha=0.4, beta=0.3, PageRank, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0.4, beta=0.3, use_pagerank=True, normalize_l2=True)
        },
        {
            'name': 'Advanced (alpha=0.5, beta=0.3, PageRank, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0.5, beta=0.3, use_pagerank=True, normalize_l2=True)
        },
        {
            'name': 'Advanced (alpha=0.6, beta=0.4, PageRank, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0.6, beta=0.4, use_pagerank=True, normalize_l2=True)
        },
        {
            'name': 'Unidirectional (alpha=0, beta=0.6)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0, beta=0.6)
        },
        {
            'name': 'Unidirectional (alpha=0, beta=0.8)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0, beta=0.8)
        },
        {
            'name': 'Unidirectional (alpha=0, beta=1.0)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0, beta=1.0)
        },
        {
            'name': 'Uni Advanced (alpha=0, beta=0.6, PR, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0, beta=0.6, use_pagerank=True, normalize_l2=True)
        },
        {
            'name': 'Uni Advanced (alpha=0, beta=0.8, PR, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0, beta=0.8, use_pagerank=True, normalize_l2=True)
        },
        {
            'name': 'Uni Advanced (alpha=0, beta=1.0, PR, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0, beta=1.0, use_pagerank=True, normalize_l2=True)
        },
    ]
    
    results = []
    # On teste toutes les config dans configurations pour trouver la meilleur
    for config in configurations:
        try:
            result = evaluate_config(queries, corpus, qrels_valid, config['func'], config['name'])
            results.append(result)
        except Exception as e:
            print(f"Erreur pour {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print("RÉSULTATS COMPARATIFS")
    
    df = pd.DataFrame(results)
    df = df.sort_values('auc_kaggle', ascending=False) # Tri par AUC Kaggle !
    print("\n" + df.to_string(index=False))
    
    # Best configuration
    best = df.iloc[0]
    print("MEILLEURE CONFIGURATION (Selon AUC Kaggle)")
    print(f"Configuration: {best['config']}")
    print(f"F1 Score:      {best['f1']:.4f}")
    print(f"AUC (Binaire):  {best['auc_kaggle']:.4f}")
    print(f"AUC (Prob):    {best['auc_prob']:.4f}")
    print(f"Precision:     {best['precision']:.4f}")
    print(f"Recall:        {best['recall']:.4f}")
    
    # Save results
if __name__ == "__main__":
    main()
