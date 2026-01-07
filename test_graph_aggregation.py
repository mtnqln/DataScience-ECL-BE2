#!/usr/bin/env python3
"""
Script pour tester différentes configurations d'agrégation de graphe
et identifier les meilleures pour améliorer les performances sur Kaggle.
"""

import pandas as pd
import numpy as np
from src.handle_data import load_corpus, load_queries, load_qrels
from src.prepare_data import embeddings_dense, embedding_query_dense
from src.model_graph import build_graph, aggregate_neighbors_advanced
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score


def evaluate_graph_config(queries, corpus, valid, embeddings_func, config_name, config):
    """Évalue une configuration d'agrégation de graphe."""
    
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"{'='*70}")
    
    # Embeddings de base
    embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    
    # Construction du graphe
    g = build_graph(corpus=corpus)
    corpus_ids = list(corpus.keys())
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    
    # Application de l'amélioration avec la config
    embeddings = embeddings_func(embeddings, g, id_to_index, **config)
    
    # Évaluation sur valid
    all_true_labels = []
    all_pred_labels = []
    all_pred_continuous_labels = []
    
    for query_id in valid.keys():
        query = queries[query_id]
        query_text = query['text']
        
        # Embedding de la requête
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings[query_idx:query_idx+1]
        else:
            query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        
        # Candidats
        query_candidates_id = list(valid[query_id].keys())
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings[candidate_index]
        
        # Similarité
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        # Top 5
        scores_sorted = np.sort(scores)[::-1]
        best_scores = scores_sorted[:5]
        
        # Labels
        all_true_labels.extend([valid[query_id][candidate_id] for candidate_id in query_candidates_id])
        all_pred_continuous_labels.extend(scores)
        all_pred_labels.extend([1 if score in best_scores else 0 for score in scores])
    
    # Métriques
    f1 = f1_score(np.array(all_true_labels), np.array(all_pred_labels), average='binary')
    auc_continuous = roc_auc_score(np.array(all_true_labels), np.array(all_pred_continuous_labels))
    auc_binary = roc_auc_score(np.array(all_true_labels), np.array(all_pred_labels))
    
    print(f"\nRésultats:")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  AUC (Prob):   {auc_continuous:.4f}")
    print(f"  AUC (Binaire): {auc_binary:.4f} ⭐")
    
    return {
        'config': config_name,
        'f1': f1,
        'auc_prob': auc_continuous,
        'auc_kaggle': auc_binary,
        **config  # Inclure les paramètres de config
    }


def main():
    print("="*70)
    print("TEST DES CONFIGURATIONS D'AGRÉGATION DE GRAPHE")
    print("="*70)
    
    # Chargement des données
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    qrels_valid = load_qrels("data/valid.tsv")
    
    print(f"\nDonnées chargées:")
    print(f"  - {len(corpus)} documents")
    print(f"  - {len(queries)} requêtes")
    print(f"  - {len(qrels_valid)} requêtes de validation")
    
    # Définir les configurations à tester
    configurations = [
        # === Baseline ===
        {
            'name': 'Baseline: Mean, Both, Uniform',
            'config': {
                'method': 'mean',
                'direction': 'both',
                'depth': 1,
                'alpha': 0.5,
                'beta': 0.3,
                'weight_metric': 'uniform',
                'normalize_l2': True
            }
        },
        
        # === Tester les méthodes d'agrégation ===
        {
            'name': 'Max, Both, Uniform',
            'config': {
                'method': 'max',
                'direction': 'both',
                'depth': 1,
                'alpha': 0.5,
                'beta': 0.3,
                'weight_metric': 'uniform',
                'normalize_l2': True
            }
        },
        {
            'name': 'Attention, Both, Uniform',
            'config': {
                'method': 'attention',
                'direction': 'both',
                'depth': 1,
                'alpha': 0.5,
                'beta': 0.3,
                'weight_metric': 'uniform',
                'normalize_l2': True
            }
        },
        
        # === Tester les métriques de poids ===
        {
            'name': 'Weighted Mean, Both, PageRank',
            'config': {
                'method': 'weighted_mean',
                'direction': 'both',
                'depth': 1,
                'alpha': 0.5,
                'beta': 0.3,
                'weight_metric': 'pagerank',
                'normalize_l2': True
            }
        },
        {
            'name': 'Weighted Mean, Both, Degree',
            'config': {
                'method': 'weighted_mean',
                'direction': 'both',
                'depth': 1,
                'alpha': 0.5,
                'beta': 0.3,
                'weight_metric': 'degree',
                'normalize_l2': True
            }
        },
        
        # === Tester les directions ===
        {
            'name': 'Weighted Mean, Out Only, PageRank',
            'config': {
                'method': 'weighted_mean',
                'direction': 'out',
                'depth': 1,
                'alpha': 0.6,
                'beta': 0.0,
                'weight_metric': 'pagerank',
                'normalize_l2': True
            }
        },
        {
            'name': 'Weighted Mean, In Only, PageRank',
            'config': {
                'method': 'weighted_mean',
                'direction': 'in',
                'depth': 1,
                'alpha': 0.0,
                'beta': 0.6,
                'weight_metric': 'pagerank',
                'normalize_l2': True
            }
        },
        
        # === Tester la profondeur ===
        {
            'name': 'Weighted Mean, Both, PageRank, Depth=2',
            'config': {
                'method': 'weighted_mean',
                'direction': 'both',
                'depth': 2,
                'alpha': 0.4,
                'beta': 0.2,
                'weight_metric': 'pagerank',
                'normalize_l2': True
            }
        },
        
        # === Combinaisons prometteuses ===
        {
            'name': 'Attention, Out Only, Uniform',
            'config': {
                'method': 'attention',
                'direction': 'out',
                'depth': 1,
                'alpha': 0.6,
                'beta': 0.0,
                'weight_metric': 'uniform',
                'normalize_l2': True
            }
        },
        {
            'name': 'Max, Out Only, PageRank',
            'config': {
                'method': 'max',
                'direction': 'out',
                'depth': 1,
                'alpha': 0.5,
                'beta': 0.0,
                'weight_metric': 'pagerank',
                'normalize_l2': True
            }
        },
    ]
    
    results = []
    
    for config_dict in configurations:
        try:
            result = evaluate_graph_config(
                queries=queries,
                corpus=corpus,
                valid=qrels_valid,
                embeddings_func=aggregate_neighbors_advanced,
                config_name=config_dict['name'],
                config=config_dict['config']
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Erreur pour {config_dict['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Afficher les résultats comparatifs
    print("\n" + "="*70)
    print("RÉSULTATS COMPARATIFS (Triés par AUC Kaggle)")
    print("="*70)
    
    df = pd.DataFrame(results)
    df = df.sort_values('auc_kaggle', ascending=False)
    
    # Afficher colonnes essentielles
    display_cols = ['config', 'auc_kaggle', 'auc_prob', 'f1', 'method', 'direction', 'weight_metric', 'depth']
    available_cols = [col for col in display_cols if col in df.columns]
    print("\n" + df[available_cols].to_string(index=False))
    
    # Top 3
    print("\n" + "="*70)
    print("🏆 TOP 3 CONFIGURATIONS")
    print("="*70)
    
    for i, row in df.head(3).iterrows():
        print(f"\n#{df.index.get_loc(i) + 1}. {row['config']}")
        print(f"   AUC (Binaire): {row['auc_kaggle']:.4f}")
        print(f"   F1 Score:     {row['f1']:.4f}")
        print(f"   Paramètres:")
        print(f"     - method: {row.get('method', 'N/A')}")
        print(f"     - direction: {row.get('direction', 'N/A')}")
        print(f"     - weight_metric: {row.get('weight_metric', 'N/A')}")
        print(f"     - depth: {row.get('depth', 'N/A')}")
        print(f"     - alpha: {row.get('alpha', 'N/A')}, beta: {row.get('beta', 'N/A')}")
    
    # Sauvegarder les résultats
    df.to_csv('results_graph_aggregation_test.csv', index=False)
    print(f"\n✓ Résultats sauvegardés dans 'results_graph_aggregation_test.csv'")


if __name__ == "__main__":
    main()
