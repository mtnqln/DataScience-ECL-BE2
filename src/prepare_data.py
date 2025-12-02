from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer

from handle_data import load_corpus
from tools import print_feats, display_side_by_side


def prepare_for_vectorizer(corpus):

    titles = [data['title'] for data in corpus.values()]
    text = [data['text'] for data in corpus.values()]
    for i in range(len(titles)):
        titles[i] = titles[i] + text[i]
    return titles

def vectorize_data(corpus_text):
    # Supprime les mots trop fréquents (and, of, what...)
    stop_words_list = list(ENGLISH_STOP_WORDS)

    # Pondération TF
    vectorizer = CountVectorizer(stop_words=stop_words_list)
    # Pondération TFxIDF
    # vectorizer = TfidfVectorizer(stop_words=stop_words_list)

    matrix = vectorizer.fit_transform(corpus_text)
    return matrix, vectorizer

def vectorize_with_sentence_transformer(corpus_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    matrix = model.encode(corpus_text)
    return matrix, model


def cosine_similarity_matrix(matrix):
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix

if __name__=="__main__":

    corpus = load_corpus("data/corpus.jsonl")
    corpus_text = prepare_for_vectorizer(corpus)
    matrix, vectorizer = vectorize_data(corpus_text)
    print(matrix[0])
    # print(matrix[1])
    # print(matrix)

    voc_1 = print_feats(matrix[0], vectorizer.get_feature_names_out(), top_n = 50)
    voc_2 = print_feats(matrix[1], vectorizer.get_feature_names_out(), top_n = 50)

    # print(voc_1)
    # print(voc_2)

    # similarity_matrix = cosine_similarity_matrix(matrix)
    # print(similarity_matrix)





    # La somme des occurrences par colonne (axis=0) donne la fréquence totale de chaque mot
    total_counts_vector = matrix.sum(axis=0)

    # Création du DataFrame de distribution (Top 30 mots)
    word_distribution_df = print_feats(
        total_counts_vector, 
        vectorizer.get_feature_names_out(), 
        top_n=30
    )

    print("\n--- 2. Distribution des 30 Mots les Plus Fréquents ---")
    print(word_distribution_df)

    # ---------------------------------------------------------------------
    # AFFICHAGE DE L'HISTOGRAMME
    # ---------------------------------------------------------------------
    
    print("\n--- 3. Génération de l'Histogramme (word_frequency_distribution.png) ---")
    
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
    plt.show()