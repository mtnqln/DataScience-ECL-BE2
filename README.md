# Moteur de Recherche Scientifique - Projet MOD 7.2

## 📋 Description du Projet

Moteur de recherche sémantique dans la littérature scientifique. Étant donné un article (requête), le système retourne les 5 articles les plus proches sémantiquement parmi ~30 candidats.

**Cours**: MOD 7.2 - Introduction à la science des données  
**Enseignants**: Julien Velcin (CM, BE), Erwan Versmée (BE)

## 🎯 Objectif

Pour chaque requête (article), identifier les 5 citations pertinentes parmi:
- 5 exemples positifs (vraies citations)
- ~25 exemples négatifs (articles aléatoires)

## 🚀 Quick Start

### Installation

```bash
cd /home/pe/code/ecl/be2_ds/DataScience-ECL-BE2

# Activer l'environnement virtuel (si nécessaire)
source .venv/bin/activate
```

### Tester les Imports

```bash
python test_imports.py
```

### Comparer les 3 Approches

```bash
python compare_approaches.py
```

### Générer les Prédictions

```bash
python generate_predictions.py
```

## 📊 Les 3 Approches Implémentées

Conformément aux exigences du projet (section 9):

| # | Approche | Description | Fichier |
|---|----------|-------------|---------|
| 1️⃣ | **Creuse** | TF-IDF avec preprocessing | [`src/prepare_data.py`](src/prepare_data.py) |
| 2️⃣ | **Dense** | Sentence Transformers + LDA | [`src/prepare_data.py`](src/prepare_data.py) |
| 3️⃣ | **Graphe** | Dense + Graphe de citations | [`src/model_graph.py`](src/model_graph.py) |

## 🔧 Corrections Apportées

### Bug Critique Corrigé

Le mapping `id_to_index` dans `validate_graph_model` était **inversé**, rendant l'amélioration par graphe inefficace.

**Avant (BUGUÉ):**
```python
id_to_index = {id:index for id,index in enumerate(corpus.keys())}
# Créait {0: 'doc1', 1: 'doc2', ...} ❌
```

**Après (CORRIGÉ):**
```python
corpus_ids = list(corpus.keys())
id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
# Crée {'doc1': 0, 'doc2': 1, ...} ✅
```

### Améliorations Algorithmiques

1. **Utilisation bidirectionnelle du graphe**
   - Avant: Seulement les références (successors)
   - Après: References + Citations (successors + predecessors)

2. **Pondération configurable**
   - α=0.3 pour les articles cités
   - β=0.2 pour les articles citants
   - Normalisation pour maintenir l'échelle

3. **Amélioration des embeddings de requêtes**
   - Si la requête est dans le corpus, son embedding est aussi amélioré

## 📁 Structure du Projet

```
DataScience-ECL-BE2/
├── src/
│   ├── handle_data.py          # Chargement des données
│   ├── prepare_data.py         # Embeddings (creux, dense, LDA)
│   ├── model_graph.py          # Graphe de citations
│   ├── model_validation.py     # Validation et prédictions
│   ├── model.py                # Moteur de recherche
│   ├── tools.py                # Utilitaires d'affichage
│   └── utils.py                # Cache et utilitaires
├── data/
│   ├── corpus.jsonl            # 25k+ articles
│   ├── queries.jsonl           # Requêtes
│   ├── valid.tsv               # Données de validation
│   ├── sample_submission.csv   # Format de soumission
│   └── sample_submission_predicted.csv  # Prédictions générées
├── test_imports.py             # Test des imports
├── test_graph_corrections.py   # Test des corrections
├── compare_approaches.py       # Comparaison des 3 approches
├── generate_predictions.py     # Génération des prédictions
├── CORRECTIONS_GRAPHE.md       # Documentation des corrections
└── README.md                   # Ce fichier
```

## 🧪 Scripts Disponibles

| Script | Description | Usage |
|--------|-------------|-------|
| **`test_imports.py`** | Vérifie que tous les imports fonctionnent | `python test_imports.py` |
| **`test_graph_corrections.py`** | Teste les corrections du modèle graphe | `python test_graph_corrections.py` |
| **`compare_approaches.py`** | Compare les 3 approches (creux, dense, graphe) | `python compare_approaches.py` |
| **`generate_predictions.py`** | Génère `sample_submission_predicted.csv` | `python generate_predictions.py` |

## 📈 Métriques d'Évaluation

Le projet utilise 4 métriques principales:

- **Precision**: Proportion de prédictions positives correctes
- **Recall**: Proportion de vrais positifs détectés
- **F1-Score**: Moyenne harmonique de Precision et Recall
- **AUC**: Aire sous la courbe ROC

## 🎓 Conformité avec les Exigences

✅ **Section 9 - Minimum Attendu:**
- ✅ 1 méthode creuse (TF-IDF)
- ✅ 1 méthode dense (Sentence Transformers + LDA)
- ✅ 1 méthode utilisant la structure (Graphe de citations)

✅ **Bonus:**
- ✅ Plusieurs variantes testées (preprocessing, pondération)
- ✅ Combinaison dense + LDA
- ✅ Amélioration bidirectionnelle du graphe
- ✅ Pondération configurable (alpha, beta)

## 📖 Documentation

- **[CORRECTIONS_GRAPHE.md](CORRECTIONS_GRAPHE.md)** - Détails des corrections du modèle graphe
- **[Guide des Prédictions](guide_predictions.md)** - Comment générer les prédictions
- **[Guide des Imports](import_fixes.md)** - Résolution des problèmes d'imports

## 🔬 Détails Techniques

### Modèle Dense + LDA

```python
# Embeddings Sentence Transformers (384 dimensions)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(corpus_text)

# Features LDA (20 topics)
lda_features = get_lda_features(corpus_text)

# Concaténation (404 dimensions au total)
final_embeddings = np.concatenate((embeddings, lda_features), axis=1)
```

### Amélioration par Graphe

```python
# Formule d'amélioration
embedding_amélioré = (
    embedding_original + 
    α * mean(embeddings_articles_cités) + 
    β * mean(embeddings_articles_citants)
) / (1 + α + β)
```

**Paramètres:**
- α = 0.3 (poids des références)
- β = 0.2 (poids des citations)

## 🐛 Dépannage

### Problèmes d'Imports

```bash
python test_imports.py
```

Si des erreurs persistent, vérifiez que tous les fichiers dans `src/` utilisent des imports relatifs (`.`).

### Mémoire Insuffisante

Si vous manquez de RAM, utilisez l'approche creuse:

```python
from src.model_validation import validate_model
validate_model(queries, corpus, qrels_valid, model_type='creux')
```

## 👥 Auteurs

Projet réalisé dans le cadre du cours MOD 7.2 - Introduction à la science des données.

## 📝 License

Projet académique - ECL 2024-2025

---

✨ **Pour toute question, consultez la documentation dans les fichiers `.md` ou contactez les enseignants.**
