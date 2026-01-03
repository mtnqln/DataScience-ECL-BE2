import torch
import numpy as np
import pandas as pd
from src.handle_data import load_corpus, load_queries, load_qrels
from src.prepare_data import embeddings_dense, embedding_query_dense
from src.model_graph import build_graph
from src.GCN_model import SimpleGCN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score

def run_gcn_pipeline():
    print("--- Démarrage de la pipeline GCN (Optimisation Multi-sauts) ---")
    
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    valid = load_qrels("data/valid.tsv")
    
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
    
    # Initialisation GCN
    gcn = SimpleGCN(input_dim=features.shape[1], hidden_dim=features.shape[1]).to(device)
    
    # Construction matrice adjacence
    adj = gcn.get_adjacency_matrix(G, corpus_ids).to(device)
    
    # Propagation (Forward Pass)
    k_hops = 3
    print(f"Propagation des features sur {k_hops} sauts...")
    
    with torch.no_grad():
        features_smoothed = gcn(features, adj, k=k_hops)
    
    embeddings_final = features_smoothed.cpu().numpy()
    
    from sklearn.preprocessing import normalize
    embeddings_final = normalize(embeddings_final, norm='l2', axis=1)
    
    print("Génération des prédictions")
    
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    
    all_true_labels = []
    all_pred_continuous = []
    all_pred_labels = []
    
    for query_id in valid.keys():
        query = queries[query_id]
        query_text = query['text']
        
        # Pour la requête, on commence par son embedding dense
        # Si la requête est DANS le graphe, on prend son embedding lissé !
        # C'est un avantage majeur du GCN.
        
        if query_id in id_to_index:
            query_idx = id_to_index[query_id]
            query_vector = embeddings_final[query_idx:query_idx+1]
        else:
            # Sinon (cas rare/impossible dans valid?), on prend le dense pur
            query_vector = embedding_query_dense(query_text, None, None, embedding_model)
            # Et on pourrait potentiellement le lisser avec ses voisins connus... 
        
        # Candidats
        candidates_ids = list(valid[query_id].keys())
        # Filtrer ceux qui sont dans le corpus (par sécurité)
        candidates_ids = [cid for cid in candidates_ids if cid in id_to_index]
        
        if not candidates_ids:
            continue
            
        cand_indices = [id_to_index[cid] for cid in candidates_ids]
        cand_vectors = embeddings_final[cand_indices]
        
        # Similarité
        sims = cosine_similarity(query_vector, cand_vectors).flatten()
        
        # Ranking
        top5_indices = np.argsort(sims)[-5:] # indices dans sims/candidates_ids
        top5_ids = [candidates_ids[i] for i in top5_indices]
        
        # Metrics accumulation
        for i, cid in enumerate(candidates_ids):
            true_label = valid[query_id][cid]
            score = sims[i]
            pred_label = 1 if cid in top5_ids else 0
            
            all_true_labels.append(true_label)
            all_pred_continuous.append(score)
            all_pred_labels.append(pred_label)
            
           
    # Calcul des métriques
    f1 = f1_score(all_true_labels, all_pred_labels)
    auc = roc_auc_score(all_true_labels, all_pred_continuous)
    
    print("\n" + "="*50)
    print(f"RÉSULTATS GCN (k={k_hops})")
    print("="*50)
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    
    # 7. Génération du fichier de soumission basé sur sample_submission.csv
    print("\nGénération du fichier de soumission 'data/sample_submission_gcn.csv'...")
    try:
        submission_df = pd.read_csv("data/sample_submission.csv")
        print(f"Chargé sample_submission.csv avec {len(submission_df)} lignes")
    except FileNotFoundError:
        print("Erreur: data/sample_submission.csv non trouvé. Impossible de générer la soumission kaggle.")
        return f1, auc

    
    queries_ids = submission_df['query-id'].unique()
    print(f"Prédictions pour {len(queries_ids)} requêtes uniques dans le fichier de soumission...")
    
    for i, qid in enumerate(queries_ids):
        if str(qid) in queries:
            query_text = queries[str(qid)]['text']
            
            if qid in id_to_index:
                 q_idx = id_to_index[qid]
                 q_vec = embeddings_final[q_idx:q_idx+1]
            else:
                 q_vec = embedding_query_dense(query_text, None, None, embedding_model)
        else:
             continue
             
        mask = submission_df['query-id'] == qid
        candidates = submission_df[mask]['corpus-id'].tolist()
        
        valid_cands_idx = []
        valid_cands_pos = []
        
        for pos, cid in enumerate(candidates):
            if cid in id_to_index:
                valid_cands_idx.append(id_to_index[cid])
                valid_cands_pos.append(pos)
                
        if not valid_cands_idx:
            continue
            
        cand_vecs = embeddings_final[valid_cands_idx]
        
        sims = cosine_similarity(q_vec, cand_vecs).flatten()
        
        
        sorted_indices_local = np.argsort(sims)[::-1]
        
        top5_local_indices = sorted_indices_local[:5]
        
        top5_positions = [valid_cands_pos[idx] for idx in top5_local_indices]
        
        row_indices = submission_df[mask].index
        
        submission_df.loc[row_indices, 'score'] = 0
        
        for pos in top5_positions:
            submission_df.loc[row_indices[pos], 'score'] = 1

    output_file = "data/sample_submission_gcn.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Fichier sauvegardé avec succès: {output_file}")
    
    return f1, auc

if __name__ == "__main__":
    run_gcn_pipeline()
