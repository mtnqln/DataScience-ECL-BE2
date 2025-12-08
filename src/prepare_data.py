from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer

from handle_data import load_corpus
from tools import print_feats, display_side_by_side


def prepare_for_vectorizer(corpus):
    titles = [data['title'] for data in corpus.values()]
    text = [data['text'] for data in corpus.values()]
    for i in range(len(titles)):
        titles[i] = titles[i] + ' ' + text[i]
    return titles

def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    stop_words_set = set(ENGLISH_STOP_WORDS)
    data_words_nostops = [[word for word in doc if word not in stop_words_set] for doc in texts]
    return data_words_nostops

def get_lda_features(corpus_text):
    """Calcule la distribution de sujets (features LDA) pour une liste de textes."""    
    lda_features = []

    corpus_text_processed = list(sent_to_words(corpus_text))
    corpus_text_processed = remove_stopwords(corpus_text_processed)

    if "ldamodel.gensim" in os.listdir("data") and "dico.dict" in os.listdir("data"):
        print("Loading existing LDA model...")
        ldamodel = LdaModel.load("data/ldamodel.gensim")
        dico = Dictionary.load("data/dico.dict")

    else:
        dico = Dictionary(corpus_text_processed)
        dico.filter_extremes(no_below=10)
        bow_corpus = [dico.doc2bow(doc) for doc in corpus_text_processed]
        # corpus = [dico.doc2bow(text) for text in data_words_nostops]

        ldamodel = LdaModel(
            bow_corpus, 
            num_topics=20,
            id2word=dico, 
            passes=100)
    
        ldamodel.save("data/ldamodel.gensim")
        dico.save("data/dico.dict")
    
    for text in corpus_text_processed:
        bow = dico.doc2bow(text)
        
        doc_topics = ldamodel.get_document_topics(bow, minimum_probability=0)
        
        vector = np.zeros(ldamodel.num_topics)
        for topic_id, prob in doc_topics:
            vector[topic_id] = prob
        
        lda_features.append(vector)
    return np.array(lda_features), ldamodel, dico


def embeddings_creux(corpus):
    corpus_text = prepare_for_vectorizer(corpus)
    corpus_text_processed = list(sent_to_words(corpus_text))
    corpus_text_processed = remove_stopwords(corpus_text_processed)

    # Pondération TF
    # model = CountVectorizer()
    # Pondération TFxIDF
    model = TfidfVectorizer()
    matrix = model.fit_transform(corpus_text_processed)

    return matrix, model

def embedding_query_dense(query_text, dico, ldamodel, embedding_model):
    '''Calcule le vecteur d'une requête en utilisant le modèle dense avec features LDA.'''
    query_vector_dense = embedding_model.encode([query_text])
    query_tokens = list(simple_preprocess(query_text, deacc=True))
    bow_query = dico.doc2bow(query_tokens)
    
    doc_topics = ldamodel.get_document_topics(bow_query, minimum_probability=0)
    
    query_vector_lda = np.zeros((1, ldamodel.num_topics))
    for topic_id, prob in doc_topics:
        query_vector_lda[0, topic_id] = prob
        
    query_vector = np.concatenate((query_vector_dense, query_vector_lda), axis=1)
    return query_vector


def embeddings_dense(corpus):
    '''Calcule les embeddings d'un corpus en utilisant un modèle dense avec features LDA.'''
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    if "embeddings_2.npy" not in os.listdir("data"):
        corpus_text = prepare_for_vectorizer(corpus)
        print("Loading dense embeddings...")
        embeddings = embedding_model.encode(corpus_text)
        # embeddings = np.load("data/embeddings.npy")

        print("Calculating LDA features...")
        lda_features, lda_model, dico = get_lda_features(corpus_text)
        embeddings_2 = np.concatenate((embeddings, lda_features), axis=1)
        print("Saving embeddings with LDA features...")

        np.save("data/embeddings_2.npy", embeddings_2)
    else:
        corpus_text = prepare_for_vectorizer(corpus)
        lda_features, lda_model, dico = get_lda_features(corpus_text)
        embeddings_2 = np.load("data/embeddings_2.npy")

    return embeddings_2, dico, lda_model, embedding_model


def cosine_similarity_matrix(matrix):
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix

if __name__=="__main__":

    corpus = load_corpus("data/corpus.jsonl")
    corpus_text = prepare_for_vectorizer(corpus)
    matrix, model = embeddings_creux(corpus)
    print(matrix[0])
    # print(matrix[1])
    # print(matrix)

    voc_1 = print_feats(matrix[0], model.get_feature_names_out(), top_n = 50)
    voc_2 = print_feats(matrix[1], model.get_feature_names_out(), top_n = 50)

    # print(voc_1)
    # print(voc_2)

    # similarity_matrix = cosine_similarity_matrix(matrix)
    # print(similarity_matrix)





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