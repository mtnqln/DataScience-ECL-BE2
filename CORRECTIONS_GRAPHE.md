# Corrections du Modèle de Validation Basé sur les Graphes

## 📋 Résumé

J'ai corrigé et amélioré le code `validate_graph_model` pour mieux répondre aux exigences du projet de recherche d'information dans la littérature scientifique (MOD 7.2 - BE séances 4, 5, 6).

## 🐛 Bug Critique Corrigé

### Problème Principal: Mapping `id_to_index` Incorrect

**Code Original (BUGUÉ):**
```python
id_to_index = {id:index for id,index in enumerate(corpus.keys())}
```

**Problème:** Cette ligne créait un dictionnaire où:
- **Clés**: Indices numériques (0, 1, 2, ...)
- **Valeurs**: IDs de documents

C'est l'inverse de ce qui est nécessaire!

**Code Corrigé:**
```python
corpus_ids = list(corpus.keys())
id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
```

**Impact:** Ce bug rendait l'amélioration des embeddings par le graphe **totalement inefficace** car les documents n'étaient jamais trouvés dans le mapping.

## 🚀 Améliorations Implémentées

### 1. Utilisation Bidirectionnelle du Graphe de Citations

**Avant:** Seules les références (successors) étaient utilisées
**Après:** Utilisation des deux directions:

```python
# Articles cités par ce document (references)
cited_papers = list(G.successors(doc_id))

# Articles qui citent ce document (cited_by)  
citing_papers = list(G.predecessors(doc_id))
```

**Justification:**
- **References**: Un article cite des travaux similaires → similarité thématique
- **Citations**: Un article cité par d'autres partage leurs thématiques → validation de pertinence

### 2. Pondération Configurable

```python
def improve_embedding(corpus_embeddings, G, id_to_index, alpha=0.3, beta=0.2):
```

- **alpha (0.3)**: Poids pour les articles cités (references)
- **beta (0.2)**: Poids pour les articles citants (cited_by)
- **Normalisation**: Division par le poids total pour maintenir l'échelle

**Formule:**
```
embedding_amélioré = (embedding_original + alpha*mean(cited) + beta*mean(citing)) / (1 + alpha + beta)
```

### 3. Amélioration des Embeddings de Requêtes

Si la requête est dans le corpus, on utilise son embedding amélioré par le graphe:

```python
if query_id in id_to_index:
    query_idx = id_to_index[query_id]
    query_vector = embeddings[query_idx:query_idx+1]
```

### 4. Meilleure Observabilité

- Statistiques détaillées sur les améliorations
- Affichage formaté des résultats
- Retour d'un dictionnaire de métriques

## 📊 Conformité avec les Exigences du Projet

Le projet (section 9) demande de tester **3 approches**:

| Approche | Description | Implémentation |
|----------|-------------|----------------|
| ✅ **Creuse** | TF-IDF | `validate_model(model_type='creux')` |
| ✅ **Dense** | Sentence Transformers + LDA | `validate_model(model_type='dense')` |
| ✅ **Structure** | Graphe de citations | `validate_graph_model()` |

## 🎯 Utilisation

### Test Rapide

```bash
cd /home/pe/code/ecl/be2_ds/DataScience-ECL-BE2
python test_graph_corrections.py
```

### Comparaison des 3 Approches

```bash
python compare_approaches.py
```

### Utilisation Programmatique

```python
from handle_data import load_corpus, load_queries, load_qrels
from model_validation import validate_graph_model

# Charger les données
corpus = load_corpus("data/corpus.jsonl")
queries = load_queries("data/queries.jsonl")
qrels_valid = load_qrels("data/valid.tsv")

# Valider le modèle avec graphe
results = validate_graph_model(queries, corpus, qrels_valid, model_type='dense')

print(f"F1-Score: {results['f1']:.4f}")
print(f"AUC: {results['auc']:.4f}")
```

## 📁 Fichiers Modifiés

### [`src/model_validation.py`](file:///wsl.localhost/Ubuntu/home/pe/code/ecl/be2_ds/DataScience-ECL-BE2/src/model_validation.py)
- ✅ Correction du mapping `id_to_index`
- ✅ Amélioration de l'embedding des requêtes
- ✅ Retour de métriques structurées
- ✅ Meilleur affichage des résultats

### [`src/model_graph.py`](file:///wsl.localhost/Ubuntu/home/pe/code/ecl/be2_ds/DataScience-ECL-BE2/src/model_graph.py)
- ✅ Utilisation bidirectionnelle du graphe
- ✅ Pondération configurable (alpha, beta)
- ✅ Normalisation des embeddings
- ✅ Statistiques détaillées

### Nouveaux Fichiers

- [`test_graph_corrections.py`](file:///wsl.localhost/Ubuntu/home/pe/code/ecl/be2_ds/DataScience-ECL-BE2/test_graph_corrections.py): Script de test
- [`compare_approaches.py`](file:///wsl.localhost/Ubuntu/home/pe/code/ecl/be2_ds/DataScience-ECL-BE2/compare_approaches.py): Comparaison des 3 approches

## 🔬 Pistes d'Amélioration Future

1. **Optimisation des hyperparamètres**: Grid search sur alpha et beta
2. **Pondération par PageRank**: Donner plus de poids aux articles influents
3. **Propagation multi-sauts**: Considérer les voisins à distance 2 ou 3
4. **Combinaison hybride**: Fusionner scores dense et graphe
5. **Graph Neural Networks**: GCN ou GAT pour apprentissage end-to-end

## 📈 Résultats Attendus

Avec ces corrections, vous devriez observer:

- ✅ **Amélioration des métriques** par rapport au modèle dense seul
- ✅ **Meilleure précision** pour les articles fortement connectés
- ✅ **Meilleur rappel** grâce à l'information structurelle

## 🎓 Contexte du Projet

**Cours**: MOD 7.2 - Introduction à la science des données  
**Enseignants**: Julien Velcin (CM, BE), Erwan Versmée (BE)  
**Objectif**: Moteur de recherche sémantique dans la littérature scientifique

**Tâche**: Pour une requête (article), retourner les 5 articles les plus proches sémantiquement parmi ~30 candidats (5 positifs + 25 négatifs).

## ✨ Conclusion

Ces modifications transforment un code **non fonctionnel** en une implémentation **robuste et conforme** aux exigences du projet. L'approche bidirectionnelle du graphe de citations devrait améliorer significativement les performances en capturant mieux les relations sémantiques entre articles scientifiques.
