import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from src.handle_data import load_corpus
from src.prepare_data import prepare_for_vectorizer, embeddings_creux, cosine_similarity_matrix
from src.tools import print_feats

if __name__=="__main__":

    print("Chargement du corpus...")
    corpus = load_corpus("data/corpus.jsonl")
    
    # print("Préparation du texte...")
    # corpus_text = prepare_for_vectorizer(corpus)
    
    print("Calcul des embeddings creux (TF-IDF)...")
    matrix, model = embeddings_creux(corpus)
    print(f"Taille de la matrice: {matrix.shape}")
    print(f"Premier vecteur (indices): {matrix[0].indices}")
    
    # Affichage des features pour le premier et le deuxième doc
    print("\n--- Features du document 0 ---")
    voc_1 = print_feats(matrix[0], model.get_feature_names_out(), top_n = 50)
    print(voc_1)
    
    print("\n--- Features du document 1 ---")
    voc_2 = print_feats(matrix[1], model.get_feature_names_out(), top_n = 50)
    print(voc_2)


    # La somme des occurrences par colonne (axis=0) donne la fréquence totale de chaque mot
    total_counts_vector = matrix.sum(axis=0)

    # Création du DataFrame de distribution (Top 30 mots)
    word_distribution_df = print_feats(
        total_counts_vector, 
        model.get_feature_names_out(), 
        top_n=30
    )

    print("\n--- 2. Distribution des 30 Mots les Plus Fréquents ---")
    print(word_distribution_df)

    # ---------------------------------------------------------------------
    # AFFICHAGE DE L'HISTOGRAMME
    # ---------------------------------------------------------------------
    
    print("\n--- 3. Génération de l'Histogramme ---")
    
    # Préparation des données pour l'histogramme
    # Inverser l'ordre pour que les barres s'affichent du plus fréquent au moins fréquent (ou vice-versa)
    words = word_distribution_df['word'].iloc[::-1]  
    counts = word_distribution_df['value'].iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(words, counts, color='teal')
    plt.xlabel("Fréquence d'apparition totale")
    plt.ylabel("Mot")
    plt.title("Distribution des 30 mots les plus fréquents (Corpus entier)")
    plt.tight_layout()
    
    output_plot = "word_frequency_distribution.png"
    plt.savefig(output_plot)
    print(f"Histogramme sauvegardé dans {output_plot}")

    try:
        plt.show()
        print("Histogramme affiché.")
    except Exception as e:
        print(f"Impossible d'afficher l'histogramme (probablement pas d'écran): {e}")