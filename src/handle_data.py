from typing import Dict
import json
import pandas as pd

def load_corpus(file_path: str) -> Dict[str, Dict]:
    """
    Load corpus data from JSONL file.
    Returns dictionary mapping document IDs to document data.
    """
    data = {}
    with open(file=file_path) as f:
        for line in f:
            json_line: dict = json.loads(line)
            id = json_line.pop("_id")
            data[str(id)] = json_line
    return data


def load_queries(file_path: str) -> Dict[str, Dict]:
    """
    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    data = {}
    with open(file=file_path) as f:
        for line in f:
            json_line: dict = json.loads(line)
            id = json_line.pop("_id")
            data[str(id)] = json_line
    return data


def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments from TSV file.
    Returns dictionary mapping query IDs to candidate relevance scores.
    """
    data = {}
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            id, docid, score = parts[0], parts[1], parts[2]
            if id not in data:
                data[id] = {}
            data[id][docid] = int(score)
    return data


if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    corpus, df = load_corpus("data/corpus.jsonl")
    first_corpus_id = list(corpus.keys())[0]
    print(corpus[first_corpus_id])
    queries = load_queries("data/queries.jsonl")
    qrels_valid = load_qrels("data/valid.tsv")

    print(f"Loaded {len(corpus)} documents in corpus")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded relevance for {len(qrels_valid)} queries (dataset)")

    # Data exploration
    # for qrel in qrels_valid.values():
    #     print(
    #         f"Proportion d'articles pertinents : {sum(qrel.values()) / len(qrel.values())}"
    #     )

    # Exemple de requete
    requete_id = list(queries.keys())[0]
    requete = queries[requete_id]
    candidats = qrels_valid[requete_id]
    print(f"Exemple de requete : \n{requete}\n et candidats : \n{candidats}\n")
