#!/usr/bin/env python3
"""
Script pour générer le fichier sample_submission_predicted.csv
en utilisant le modèle avec graphe amélioré.
"""

import pandas as pd
from src.handle_data import load_corpus, load_queries
from src.model_validation import sample_prediction_graph

def main():
    print("="*80)
    print("GÉNÉRATION DES PRÉDICTIONS AVEC MODÈLE GRAPHE")
    print("="*80)
    
    # Charger les données
    print("\n[1/4] Chargement des données...")
    try:
        corpus = load_corpus("data/corpus.jsonl")
        queries = load_queries("data/queries.jsonl")
        sample_submission = pd.read_csv("data/sample_submission.csv")
        
        print(f"  ✓ Corpus: {len(corpus)} documents")
        print(f"  ✓ Queries: {len(queries)} requêtes")
        print(f"  ✓ Sample submission: {len(sample_submission)} lignes")
        print(f"  ✓ Nombre de requêtes uniques: {len(sample_submission['query-id'].unique())}")
    except Exception as e:
        print(f"  ✗ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Générer les prédictions
    print("\n[2/4] Génération des prédictions avec modèle graphe...")
    print("  (Cela peut prendre plusieurs minutes...)")
    try:
        predictions = sample_prediction_graph(
            queries=queries,
            corpus=corpus,
            valid=sample_submission.copy(),
            model_type='dense'
        )
        print("  ✓ Prédictions générées avec succès")
    except Exception as e:
        print(f"  ✗ Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Vérifier le fichier généré
    print("\n[3/4] Vérification du fichier généré...")
    try:
        df = pd.read_csv("data/sample_submission_predicted.csv")
        print(f"  ✓ Fichier chargé: {len(df)} lignes")
        print(f"  ✓ Colonnes: {list(df.columns)}")
        
        # Statistiques
        print(f"\n  Statistiques:")
        print(f"    - Nombre de prédictions positives (score=1): {(df['score'] == 1).sum()}")
        print(f"    - Nombre de prédictions négatives (score=0): {(df['score'] == 0).sum()}")
        print(f"    - Ratio positif/négatif: {(df['score'] == 1).sum() / len(df):.2%}")
        
        # Afficher un échantillon
        print(f"\n  Échantillon des prédictions:")
        print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"  ✗ Erreur lors de la vérification: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[4/4] Résumé")
    print("="*80)
    print("✅ Fichier généré avec succès: data/sample_submission_predicted.csv")
    print(f"✅ {len(df)} prédictions générées")
    print(f"✅ Utilisation du modèle: Dense + LDA + Graphe de citations")
    print("="*80)

if __name__ == "__main__":
    main()
