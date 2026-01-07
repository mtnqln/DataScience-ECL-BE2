
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score
from src.handle_data import load_corpus, load_queries, load_qrels
from src.prepare_data import embeddings_creux, embedding_query_dense

def run_sparse_model():
    """script qui run la methode creuse"""
    
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    valid = load_qrels("data/valid.tsv")
    
    # Vectorisation du corpus (TF-IDF)
    print("Calcul des embeddings creux (TF-IDF)")
    matrix, vectorizer = embeddings_creux(corpus)
    
    corpus_ids = list(corpus.keys())
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    
    
    
    all_true_labels = []
    all_pred_continuous = []
    all_pred_labels = []
    
    submission_rows = []
    
    queries_ids = list(valid.keys())
    
    for count, qid in enumerate(queries_ids):
        if str(qid) not in queries:
            continue
            
        query_text = queries[str(qid)]['text']
        
        # Vecteur requête (transform)
        q_vec = vectorizer.transform([query_text])
        
        # Candidats
        candidates = list(valid[qid].keys())
        candidates_idx = []
        candidates_pos = []
        
        for pos, cid in enumerate(candidates):
            if cid in id_to_index:
                candidates_idx.append(id_to_index[cid])
                candidates_pos.append(pos)
                
        if not candidates_idx:
            # Fallback
            current_scores = {cid: 0 for cid in candidates}
            for cid in candidates:
                 submission_rows.append({'query-id': qid, 'corpus-id': cid, 'score': 0})
            continue
            
        cand_matrix = matrix[candidates_idx] # type: ignore
        
        sims = cosine_similarity(q_vec, cand_matrix).flatten()
        
        # Ranking
        sorted_indices_local = np.argsort(sims)[::-1]
        top5_local_indices = sorted_indices_local[:5]
        top5_positions = [candidates_pos[idx] for idx in top5_local_indices]
        
        # Metrics accumulation
        top5_ids_set = {candidates[p] for p in top5_positions}
        
        for i, idx_in_subset in enumerate(candidates_idx):
            cid = corpus_ids[idx_in_subset] 
            score = sims[i]
            true_label = valid[qid][cid]
            pred_label = 1 if cid in top5_ids_set else 0
            
            all_true_labels.append(true_label)
            all_pred_continuous.append(score)
            all_pred_labels.append(pred_label)
            
        # Submission generation - Utiliser les scores continus
        # Créer un dictionnaire des scores pour tous les candidats
        current_scores = {}
        for i, idx_in_subset in enumerate(candidates_idx):
            cid = corpus_ids[idx_in_subset] 
            current_scores[cid] = sims[i]  # Garder les valeurs de similarité cosinus
            
        # Ajouter aussi les candidats qui n'étaient pas dans l'index avec score=0
        for cid in candidates:
            if cid not in current_scores:
                current_scores[cid] = 0.0
            
        for cid in candidates:
             submission_rows.append({
                 'query-id': qid, 
                 'corpus-id': cid, 
                 'score': current_scores[cid]
             })

    # Evaluation
    f1 = f1_score(all_true_labels, all_pred_labels)
    auc_continuous = roc_auc_score(all_true_labels, all_pred_continuous)
    auc_binary = roc_auc_score(all_true_labels, all_pred_labels)
    
    print("RÉSULTATS MÉTHODE CREUSE")
    print("="*50)
    print(f"F1 Score:       {f1:.4f}")
    print(f"AUC (Prob):     {auc_continuous:.4f}")
    print(f"AUC (Binaire):   {auc_binary:.4f}")
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.insert(0, 'RowId', range(len(submission_df)))
    submission_df = submission_df[['RowId', 'query-id', 'corpus-id', 'score']]
    
    output_file = "submissions/sample_submission_sparse.csv"
    submission_df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    run_sparse_model()
