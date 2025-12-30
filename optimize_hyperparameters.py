#!/usr/bin/env python3
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
    
    print(f"\n{'='*80}")
    print(f"TESTING: {config_name}")
    print(f"{'='*80}")
    
    # Get embeddings
    embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    
    # Build graph
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
    auc = roc_auc_score(np.array(all_true_labels), np.array(all_pred_continuous_labels))
    
    print(f"\nRésultats:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    return {
        'config': config_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main():
    print("="*80)
    print("OPTIMISATION DES HYPERPARAMÈTRES")
    print("="*80)
    
    # Load data
    print("\nChargement des données...")
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    qrels_valid = load_qrels("data/valid.tsv")
    print(f"✓ {len(corpus)} documents, {len(queries)} requêtes")
    
    # Test configurations
    configurations = [
        {
            'name': 'Baseline (α=0.3, β=0.2)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.3, beta=0.2)
        },
        {
            'name': 'Higher weights (α=0.4, β=0.3)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.4, beta=0.3)
        },
        {
            'name': 'More references (α=0.5, β=0.2)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.5, beta=0.2)
        },
        {
            'name': 'More citations (α=0.3, β=0.4)',
            'func': lambda emb, g, idx: improve_embedding(emb, g, idx, alpha=0.3, beta=0.4)
        },
        {
            'name': 'Advanced (α=0.4, β=0.3, PageRank, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0.4, beta=0.3, use_pagerank=True, normalize_l2=True)
        },
        {
            'name': 'Advanced (α=0.5, β=0.3, PageRank, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0.5, beta=0.3, use_pagerank=True, normalize_l2=True)
        },
        {
            'name': 'Advanced (α=0.6, β=0.4, PageRank, L2)',
            'func': lambda emb, g, idx: improve_embedding_advanced(emb, g, idx, alpha=0.6, beta=0.4, use_pagerank=True, normalize_l2=True)
        },
    ]
    
    results = []
    
    for config in configurations:
        try:
            result = evaluate_config(queries, corpus, qrels_valid, config['func'], config['name'])
            results.append(result)
        except Exception as e:
            print(f"✗ Erreur pour {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Display results
    print("\n" + "="*80)
    print("RÉSULTATS COMPARATIFS")
    print("="*80)
    
    df = pd.DataFrame(results)
    df = df.sort_values('f1', ascending=False)
    print("\n" + df.to_string(index=False))
    
    # Best configuration
    best = df.iloc[0]
    print("\n" + "="*80)
    print("MEILLEURE CONFIGURATION")
    print("="*80)
    print(f"Configuration: {best['config']}")
    print(f"F1 Score: {best['f1']:.4f}")
    print(f"AUC: {best['auc']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    
    # Save results
    df.to_csv('data/hyperparameter_optimization_results.csv', index=False)
    print(f"\n✅ Résultats sauvegardés dans: data/hyperparameter_optimization_results.csv")

if __name__ == "__main__":
    main()
