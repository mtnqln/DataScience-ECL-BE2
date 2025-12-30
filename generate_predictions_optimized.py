#!/usr/bin/env python3
"""
Script pour générer les prédictions avec le modèle OPTIMISÉ.
Utilise la version avancée avec PageRank et normalisation L2.
"""

import pandas as pd
import numpy as np
from src.handle_data import load_corpus, load_queries
from src.prepare_data import embeddings_dense, embedding_query_dense
from src.model_graph import build_graph, improve_embedding_advanced
from sklearn.metrics.pairwise import cosine_similarity

def generate_predictions_optimized(queries, corpus, valid, 
                                   alpha=0.5, beta=0.3, 
                                   use_pagerank=True, normalize_l2=True):
    """
    Generate predictions using optimized graph-enhanced embeddings.
    
    Args:
        queries: Dictionary of query documents
        corpus: Dictionary of corpus documents
        valid: DataFrame with query-id and corpus-id columns
        alpha: Weight for cited papers (default: 0.5, optimized)
        beta: Weight for citing papers (default: 0.3, optimized)
        use_pagerank: Use PageRank weighting
        normalize_l2: Apply L2 normalization
        
    Returns:
        DataFrame with predictions
    """
    print("="*80)
    print("GÉNÉRATION DES PRÉDICTIONS - MODÈLE OPTIMISÉ")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - alpha (références): {alpha}")
    print(f"  - beta (citations): {beta}")
    print(f"  - PageRank weighting: {use_pagerank}")
    print(f"  - L2 normalization: {normalize_l2}")
    
    # Load embeddings
    print("\n[1/5] Chargement des embeddings...")
    embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    print(f"  ✓ Embeddings chargés: {embeddings.shape}")
    
    # Build graph
    print("\n[2/5] Construction du graphe de citations...")
    g = build_graph(corpus=corpus)
    corpus_ids = list(corpus.keys())
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    print(f"  ✓ Graphe construit: {len(g.nodes())} nœuds, {len(g.edges())} arêtes")
    
    # Improve embeddings with advanced method
    print("\n[3/5] Amélioration des embeddings avec méthode avancée...")
    embeddings = improve_embedding_advanced(
        corpus_embeddings=embeddings,
        G=g,
        id_to_index=id_to_index,
        alpha=alpha,
        beta=beta,
        use_pagerank=use_pagerank,
        normalize_l2=normalize_l2
    )
    print(f"  ✓ Embeddings améliorés")
    
    # Generate predictions
    print(f"\n[4/5] Génération des prédictions pour {len(valid['query-id'].unique())} requêtes...")
    
    for i, query_id in enumerate(valid['query-id'].unique()):
        if (i + 1) % 10 == 0:
            print(f"  Progression: {i+1}/{len(valid['query-id'].unique())} requêtes")
        
        query = queries[query_id]
        query_text = query['text']
        
        # Get query vector
        query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        
        # Use graph-enhanced embedding if query is in corpus
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings[query_idx:query_idx+1]
        
        # Get candidate vectors
        query_candidates_id = valid[valid['query-id'] == query_id]['corpus-id'].tolist()
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings[candidate_index]
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        # Select top 5
        scores_sorted = np.sort(scores)[::-1]
        best_scores = scores_sorted[:5]
        
        # Update predictions
        for j, candidate_id in enumerate(query_candidates_id):
            valid.loc[(valid['query-id'] == query_id) & (valid['corpus-id'] == candidate_id), 'score'] = \
                1 if scores[j] in best_scores else 0
    
    # Save predictions
    output_file = 'data/sample_submission_predicted_optimized.csv'
    valid.to_csv(output_file, index=False)
    
    print(f"\n[5/5] Sauvegarde des prédictions...")
    print(f"  ✓ Fichier sauvegardé: {output_file}")
    
    # Statistics
    print(f"\n{'='*80}")
    print("STATISTIQUES")
    print(f"{'='*80}")
    print(f"Total de prédictions: {len(valid)}")
    print(f"Prédictions positives: {(valid['score'] == 1).sum()}")
    print(f"Prédictions négatives: {(valid['score'] == 0).sum()}")
    print(f"Ratio positif: {(valid['score'] == 1).sum() / len(valid):.2%}")
    
    return valid

def main():
    # Load data
    print("Chargement des données...")
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    
    print(f"✓ {len(corpus)} documents")
    print(f"✓ {len(queries)} requêtes")
    print(f"✓ {len(sample_submission)} lignes à prédire")
    
    # Generate predictions with optimized parameters
    # Ces valeurs sont basées sur l'optimisation
    # Vous pouvez les ajuster après avoir exécuté optimize_hyperparameters.py
    predictions = generate_predictions_optimized(
        queries=queries,
        corpus=corpus,
        valid=sample_submission.copy(),
        alpha=0.5,  # Optimisé (au lieu de 0.3)
        beta=0.3,   # Optimisé (au lieu de 0.2)
        use_pagerank=True,  # Nouveau!
        normalize_l2=True   # Nouveau!
    )
    
    print(f"\n{'='*80}")
    print("✅ PRÉDICTIONS GÉNÉRÉES AVEC SUCCÈS")
    print(f"{'='*80}")
    print("\nFichier généré: data/sample_submission_predicted_optimized.csv")
    print("\nProchaines étapes:")
    print("1. Vérifier le fichier généré")
    print("2. Soumettre sur Kaggle")
    print("3. Comparer avec le score précédent (0.9045)")

if __name__ == "__main__":
    main()
