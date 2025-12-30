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
    Génération des prédictions par graphe
    On inclut des poids liés à l'algorithme de page rank vu en cours
    alpha est le poids pour les papiers cités
    beta est le poids pour les papiers citant    

    """
    print(f"alpha (références): {alpha}")
    print(f"beta (citations): {beta}")
    print(f"PageRank ?: {use_pagerank}")
    print(f"L2 normalization: {normalize_l2}")
    
    # Load embeddings
    print(" Etape 1 : Chargement des embeddings...")
    embeddings, dico, lda_model, embedding_model = embeddings_dense(corpus)
    print(f"Embeddings chargés: {embeddings.shape}")
    
    # Build graph
    print(" Etape 2 : Construction du graphe de citations")
    g = build_graph(corpus=corpus)
    corpus_ids = list(corpus.keys())
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    print(f"Graphe construit: {len(g.nodes())} noeuds, {len(g.edges())} arêtes")
    
    # Improve embeddings with advanced method
    print(" Amélioration des embeddings avec méthode avancée...")
    embeddings = improve_embedding_advanced(
        corpus_embeddings=embeddings,
        G=g,
        id_to_index=id_to_index,
        alpha=alpha,
        beta=beta,
        use_pagerank=use_pagerank,
        normalize_l2=normalize_l2
    )
    
    print(f" Dernière étape : Génération des prédictions pour {len(valid['query-id'].unique())} requêtes")
    
    for i, query_id in enumerate(valid['query-id'].unique()):
        if (i + 1) % 10 == 0:
            print(f"  Progression: {i+1}/{len(valid['query-id'].unique())} requêtes")
        
        query = queries[query_id]
        query_text = query['text']
        
        query_vector = embedding_query_dense(query_text, dico, lda_model, embedding_model)
        
        # Use graph-enhanced embedding if query is in corpus
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings[query_idx:query_idx+1]
        
        # Get candidate vectors
        query_candidates_id = valid[valid['query-id'] == query_id]['corpus-id'].tolist()
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings[candidate_index]
        
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        scores_sorted = np.sort(scores)[::-1]
        best_scores = scores_sorted[:5]
        
        # C'est ici qu'on met à jour les scores pour créer le fichier de csv
        for j, candidate_id in enumerate(query_candidates_id):
            valid.loc[(valid['query-id'] == query_id) & (valid['corpus-id'] == candidate_id), 'score'] = \
                1 if scores[j] in best_scores else 0
    

    output_file = 'data/sample_submission_predicted_optimized.csv'
    valid.to_csv(output_file, index=False)
    
    
    # Statistiques

    print(f"Total de prédictions: {len(valid)}")
    print(f"Prédictions positives: {(valid['score'] == 1).sum()}")
    print(f"Prédictions négatives: {(valid['score'] == 0).sum()}")
    print(f"Ratio positif: {(valid['score'] == 1).sum() / len(valid):.2%}")
    
    return valid

def main():
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    
    print(f"{len(corpus)} documents")
    print(f"{len(queries)} requêtes")
    print(f"{len(sample_submission)} lignes à prédire")
    
    # Generate predictions with optimized parameters
    # Ces valeurs sont basées sur l'optimisation
    predictions = generate_predictions_optimized(
        queries=queries,
        corpus=corpus,
        valid=sample_submission.copy(),
        alpha=0.5,  # Optimisé (au lieu de 0.3)
        beta=0.3,   # Optimisé (au lieu de 0.2)
        use_pagerank=True,  # Nouveau!
        normalize_l2=True   # Nouveau!
    )
    

if __name__ == "__main__":
    main()
