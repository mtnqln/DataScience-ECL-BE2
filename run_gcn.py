import torch
import numpy as np
import pandas as pd
import networkx as nx
from src.handle_data import load_corpus, load_queries, load_qrels
from src.prepare_data import embeddings_dense, embedding_query_dense
from src.model_graph import build_graph
from src.GCN_model import SimpleGCN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score

def run_gcn_pipeline():
    print("--- Démarrage de la pipeline GCN + PageRank Boost ---")
    
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    valid = load_qrels("data/valid.tsv")  # Pour l'évaluation
    sample_submission = pd.read_csv("data/sample_submission.csv")  # Template pour la soumission
    
    print("Chargement des embeddings initiaux...")
    embeddings, _, _, embedding_model = embeddings_dense(corpus) # on ignore les lda features

    
    # Passage en Tensor PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    features = torch.FloatTensor(embeddings).to(device)
    
    # Construction du Graphe
    print("Construction du graphe de citations...")
    G = build_graph(corpus)
    corpus_ids = list(corpus.keys())
    
    # Calcul du PageRank
    print("Calcul du PageRank pour le boosting...")
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    
    # Initialisation GCN
    gcn = SimpleGCN(input_dim=features.shape[1], hidden_dim=features.shape[1]).to(device)
    
    # Construction matrice adjacence
    adj = gcn.get_adjacency_matrix(G, corpus_ids).to(device)
    
    # Propagation (Forward Pass)
    k_hops = 2
    print(f"Propagation des features sur {k_hops} sauts...")
    
    with torch.no_grad():
        features_smoothed = gcn(features, adj, k=k_hops)
    
    embeddings_final = features_smoothed.cpu().numpy()
    
    from sklearn.preprocessing import normalize
    embeddings_final = normalize(embeddings_final, norm='l2', axis=1)
    
    print("Génération des prédictions (avec PageRank Boost)")
    
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    
    all_true_labels = []
    all_pred_continuous = []
    all_pred_labels = []
    
    # Paramètre de boost PageRank
    PR_BOOST_FACTOR = 100.0 # Facteur expérimental pour donner du poids au PR (les scores PR sont très petits, ex 1e-4)

    for query_id in valid.keys():
        query = queries[query_id]
        query_text = query['text']
        
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings_final[query_idx:query_idx+1]
        else:
            query_vector = embedding_query_dense(query_text, None, None, embedding_model)
        
        candidates_ids = list(valid[query_id].keys())
        candidates_ids = [cid for cid in candidates_ids if cid in id_to_index]
        
        if not candidates_ids:
            continue
            
        cand_indices = [id_to_index[cid] for cid in candidates_ids]
        cand_vectors = embeddings_final[cand_indices]
        
        # Similarité Cosinus
        sims = cosine_similarity(query_vector, cand_vectors).flatten()
        
        # Integration PageRank
        # Score final = Sim * (1 + Factor * PR)
        pr_values = np.array([pagerank_scores.get(cid, 0) for cid in candidates_ids])
        final_scores = sims * (1 + PR_BOOST_FACTOR * pr_values)
        
        # Ranking basé sur le score final fusionné
        top5_indices = np.argsort(final_scores)[-5:]
        top5_ids = [candidates_ids[i] for i in top5_indices]
        
        # Metrics accumulation
        for i, cid in enumerate(candidates_ids):
            true_label = valid[query_id][cid]
            score = final_scores[i] # Utilisation du score boosté
            pred_label = 1 if cid in top5_ids else 0
            
            all_true_labels.append(true_label)
            all_pred_continuous.append(score)
            all_pred_labels.append(pred_label)
            
           
    # Calcul des métriques
    f1 = f1_score(all_true_labels, all_pred_labels)
    auc_continuous = roc_auc_score(all_true_labels, all_pred_continuous)
    auc_binary = roc_auc_score(all_true_labels, all_pred_labels)
    
    print(f"RÉSULTATS GCN + PageRank Boost (k={k_hops})")
    print(f"F1 Score:       {f1:.4f}")
    print(f"AUC (Prob):     {auc_continuous:.4f} (Potentiel max)")
    print(f"AUC (Binaire):   {auc_binary:.4f} (Estimation sur 0/1)")
    
    print(f"\nGénération des prédictions pour {len(sample_submission['query-id'].unique())} requêtes de test...")
    
    # Parcourir chaque requête du template de soumission
    for i, query_id in enumerate(sample_submission['query-id'].unique()):
        if (i + 1) % 50 == 0:
            print(f"  Progression: {i+1}/{len(sample_submission['query-id'].unique())} requêtes")
        
        query = queries[query_id]
        query_text = query['text']
        
        # Utiliser l'embedding du graphe si la requête est dans le corpus
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings_final[query_idx:query_idx+1]
        else:
            query_vector = embedding_query_dense(query_text, None, None, embedding_model)
        
        # Récupérer les candidats pour cette requête
        query_candidates_id = sample_submission[sample_submission['query-id'] == query_id]['corpus-id'].tolist()
        candidate_index = [corpus_ids.index(candidate_id) for candidate_id in query_candidates_id]
        query_candidate_vectors = embeddings_final[candidate_index]
        
        # Calcul de similarité
        similarity_matrix = cosine_similarity(query_vector, query_candidate_vectors)
        scores = similarity_matrix.flatten()
        
        # Application du boost PageRank
        pr_values = np.array([pagerank_scores.get(cid, 0) for cid in query_candidates_id])
        final_scores = scores * (1 + PR_BOOST_FACTOR * pr_values)
        
        # Sélection des 5 meilleurs
        scores_sorted = np.sort(final_scores)[::-1]
        best_scores = scores_sorted[:5]
        
        # Mise à jour des scores dans le DataFrame de soumission
        for j, candidate_id in enumerate(query_candidates_id):
            sample_submission.loc[(sample_submission['query-id'] == query_id) & (sample_submission['corpus-id'] == candidate_id), 'score'] = \
                final_scores[j]  

    output_file = "submissions/sample_submission_gcn.csv"
    sample_submission.to_csv(output_file, index=False)
    print(f"\nFichier sauvegardé: {output_file}")
    print(f"Total de prédictions: {len(sample_submission)}")
    
    return f1, auc_binary, auc_continuous

if __name__ == "__main__":
    run_gcn_pipeline()

