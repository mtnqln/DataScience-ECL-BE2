#!/usr/bin/env python3
"""
Script pour comparer deux fichiers de prédictions.
"""

import pandas as pd
import numpy as np

def compare_predictions(file1, file2, file1_name="File 1", file2_name="File 2"):
    """Compare two prediction CSV files."""
    
    print("="*80)
    print(f"COMPARAISON: {file1_name} vs {file2_name}")
    print("="*80)
    
    # Load files
    print(f"\nChargement des fichiers...")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    print(f"  ✓ {file1_name}: {len(df1)} lignes")
    print(f"  ✓ {file2_name}: {len(df2)} lignes")
    
    # Basic comparison
    print(f"\n{'='*80}")
    print("STATISTIQUES GLOBALES")
    print(f"{'='*80}")
    
    stats = pd.DataFrame({
        file1_name: [
            len(df1),
            (df1['score'] == 1).sum(),
            (df1['score'] == 0).sum(),
            f"{(df1['score'] == 1).sum() / len(df1):.2%}"
        ],
        file2_name: [
            len(df2),
            (df2['score'] == 1).sum(),
            (df2['score'] == 0).sum(),
            f"{(df2['score'] == 1).sum() / len(df2):.2%}"
        ]
    }, index=['Total lignes', 'Prédictions positives (1)', 'Prédictions négatives (0)', 'Ratio positif'])
    
    print("\n" + stats.to_string())
    
    # Merge and compare
    print(f"\n{'='*80}")
    print("ANALYSE DES DIFFÉRENCES")
    print(f"{'='*80}")
    
    merged = df1.merge(df2, on=['query-id', 'corpus-id'], suffixes=('_1', '_2'))
    
    # Count differences
    differences = merged[merged['score_1'] != merged['score_2']]
    agreements = merged[merged['score_1'] == merged['score_2']]
    
    print(f"\nPrédictions identiques: {len(agreements)} ({len(agreements)/len(merged):.2%})")
    print(f"Prédictions différentes: {len(differences)} ({len(differences)/len(merged):.2%})")
    
    if len(differences) > 0:
        print(f"\nDétail des différences:")
        
        # 1 -> 0
        changed_1_to_0 = differences[(differences['score_1'] == 1) & (differences['score_2'] == 0)]
        print(f"  - {file1_name} prédit 1, {file2_name} prédit 0: {len(changed_1_to_0)}")
        
        # 0 -> 1
        changed_0_to_1 = differences[(differences['score_1'] == 0) & (differences['score_2'] == 1)]
        print(f"  - {file1_name} prédit 0, {file2_name} prédit 1: {len(changed_0_to_1)}")
        
        # Show some examples
        if len(differences) > 0:
            print(f"\nExemples de différences (10 premiers):")
            print(differences[['query-id', 'corpus-id', 'score_1', 'score_2']].head(10).to_string(index=False))
    
    # Per-query analysis
    print(f"\n{'='*80}")
    print("ANALYSE PAR REQUÊTE")
    print(f"{'='*80}")
    
    query_diff = merged.groupby('query-id').apply(
        lambda x: (x['score_1'] != x['score_2']).sum()
    ).sort_values(ascending=False)
    
    queries_with_diff = (query_diff > 0).sum()
    queries_identical = (query_diff == 0).sum()
    
    print(f"\nRequêtes avec prédictions identiques: {queries_identical}")
    print(f"Requêtes avec au moins une différence: {queries_with_diff}")
    
    if queries_with_diff > 0:
        print(f"\nTop 10 requêtes avec le plus de différences:")
        print(query_diff.head(10).to_string())
    
    # Summary
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print(f"{'='*80}")
    
    if len(differences) == 0:
        print("\n✅ Les deux fichiers sont IDENTIQUES!")
    else:
        print(f"\n⚠️  Les fichiers sont DIFFÉRENTS:")
        print(f"  - {len(differences)} prédictions différentes sur {len(merged)} ({len(differences)/len(merged):.2%})")
        print(f"  - {queries_with_diff} requêtes affectées sur {len(merged['query-id'].unique())}")
    
    return merged, differences

if __name__ == "__main__":
    # Compare the two files
    merged, differences = compare_predictions(
        "data/sample_submission_predicted.csv",
        "data/sample_submission_predicted_test.csv",
        file1_name="predicted",
        file2_name="predicted_test"
    )
    
    # Save differences if any
    if len(differences) > 0:
        differences.to_csv('data/prediction_differences.csv', index=False)
        print(f"\n✓ Différences sauvegardées dans: data/prediction_differences.csv")
