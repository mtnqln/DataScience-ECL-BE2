#!/usr/bin/env python3
"""
Comparaison rapide des deux fichiers de prédictions.
"""

import pandas as pd

# Load files
print("Chargement des fichiers...")
df1 = pd.read_csv("data/sample_submission_predicted.csv")
df2 = pd.read_csv("data/sample_submission_predicted_test.csv")

print(f"\n{'='*80}")
print("STATISTIQUES GLOBALES")
print(f"{'='*80}")

print(f"\nsample_submission_predicted.csv:")
print(f"  - Total: {len(df1)} lignes")
print(f"  - Positifs (1): {(df1['score'] == 1).sum()}")
print(f"  - Négatifs (0): {(df1['score'] == 0).sum()}")
print(f"  - Ratio: {(df1['score'] == 1).sum() / len(df1):.2%}")

print(f"\nsample_submission_predicted_test.csv:")
print(f"  - Total: {len(df2)} lignes")
print(f"  - Positifs (1): {(df2['score'] == 1).sum()}")
print(f"  - Négatifs (0): {(df2['score'] == 0).sum()}")
print(f"  - Ratio: {(df2['score'] == 1).sum() / len(df2):.2%}")

# Merge and compare
merged = df1.merge(df2, on=['query-id', 'corpus-id'], suffixes=('_1', '_2'))
differences = merged[merged['score_1'] != merged['score_2']]

print(f"\n{'='*80}")
print("COMPARAISON")
print(f"{'='*80}")

print(f"\nPrédictions identiques: {len(merged) - len(differences)} ({(len(merged) - len(differences))/len(merged):.2%})")
print(f"Prédictions différentes: {len(differences)} ({len(differences)/len(merged):.2%})")

if len(differences) > 0:
    changed_1_to_0 = len(differences[(differences['score_1'] == 1) & (differences['score_2'] == 0)])
    changed_0_to_1 = len(differences[(differences['score_1'] == 0) & (differences['score_2'] == 1)])
    
    print(f"\nDétail:")
    print(f"  - predicted=1, test=0: {changed_1_to_0}")
    print(f"  - predicted=0, test=1: {changed_0_to_1}")
    
    print(f"\nExemples (5 premiers):")
    print(differences[['query-id', 'corpus-id', 'score_1', 'score_2']].head(5))
else:
    print("\n✅ Les deux fichiers sont IDENTIQUES!")

print(f"\n{'='*80}")
