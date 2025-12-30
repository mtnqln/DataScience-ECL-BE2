#!/usr/bin/env python3
"""
Script simple pour tester les imports après corrections.
"""

print("="*70)
print("TEST DES IMPORTS")
print("="*70)

try:
    print("\n[1/5] Import de src.handle_data...")
    from src.handle_data import load_corpus, load_queries, load_qrels
    print("  ✓ src.handle_data importé avec succès")
except Exception as e:
    print(f"  ✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n[2/5] Import de src.prepare_data...")
    from src.prepare_data import embeddings_creux, embeddings_dense
    print("  ✓ src.prepare_data importé avec succès")
except Exception as e:
    print(f"  ✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n[3/5] Import de src.model_graph...")
    from src.model_graph import build_graph, improve_embedding
    print("  ✓ src.model_graph importé avec succès")
except Exception as e:
    print(f"  ✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n[4/5] Import de src.model_validation...")
    from src.model_validation import validate_model, validate_graph_model
    print("  ✓ src.model_validation importé avec succès")
except Exception as e:
    print(f"  ✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n[5/5] Test de chargement des données...")
    corpus = load_corpus("data/corpus.jsonl")
    print(f"  ✓ Corpus chargé: {len(corpus)} documents")
except Exception as e:
    print(f"  ✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TESTS D'IMPORTS TERMINÉS")
print("="*70)
