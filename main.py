from src.handle_data import load_corpus, load_queries
from generate_predictions_optimized import generate_predictions_optimized
import pandas as pd

# Ce code main permet de générer le fichier
def main(ouput_file=None):
    corpus = load_corpus("data/corpus.jsonl")
    queries = load_queries("data/queries.jsonl")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    
    print(f"{len(corpus)} documents")
    print(f"{len(queries)} requêtes")
    print(f"{len(sample_submission)} lignes à prédire")
    

    predictions = generate_predictions_optimized(
        queries=queries,
        corpus=corpus,
        valid=sample_submission.copy(),
        alpha=0.5,
        beta=0.3,
        use_pagerank=True,
        normalize_l2=True,
        output_file=ouput_file
    )
    
if __name__ == "__main__":
    ouput_file = input("nom du fichier d'output: ")
    main(ouput_file)
