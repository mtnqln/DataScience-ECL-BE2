PROJET MOTEUR DE RECHERCHE - README

ORGANISATION DES DOSSIERS

- data/ : Contient les donnees du projet (corpus, requetes, etc.)
- src/ : Contient tout le code source (fonctions, modeles)
- submissions/ : Contient les fichiers CSV generes pour Kaggle
- main.py : Le programme principal pour creer une soumission
- run_gcn.py : Le script pour le modele GCN (Graph Neural Network)
- requirements.txt : Liste des dependances Python

INSTALLATION DE L'ENVIRONNEMENT

**Pour installer les librairies necessaires :**

- Ouvrez un terminal
- Tapez : pip install -r requirements.txt

COMMENT UTILISER LE PROGRAMME (GENERER UNE SOUMISSION)

**Pour lancer le generateur avec les reglages par defaut :**

- Ouvrez un terminal
- Tapez : python main.py (main va utiliser generate_predictions_optimized.py)
- Le programme va vous demander le nom du fichier de sortie (ex: ma_soumission)
- Le fichier sera cree dans le dossier "submissions/"

**Pour choisir vos propres hyperparametres (Alpha, Beta, etc.) :**

- Ouvrez le fichier "main.py" avec un editeur de texte
- Cherchez les lignes "alpha=0" et "beta=1" (vers la ligne 21)
Modifiez ces valeurs selon vos besoins (ex: alpha=0.3, beta=0.5)
Sauvegardez le fichier
Lancez ensuite : python main.py

**Pour utiliser le model GCN , creux , dense :**

- Tapez dans un terminal : python run_gcn.py
- Taper dans le terminal : python run_sparse.py
- Taper dans le terminal : python run_dense.py

**Pour visualiser la distribution des mots par LDA**

- Tapez dans un terminal : python visualize_datas.py