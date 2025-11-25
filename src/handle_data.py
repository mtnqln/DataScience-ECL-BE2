from typing import Dict


def load_corpus(file_path: str) -> Dict[str, Dict]:
    """
    TODO

    Load corpus data from JSONL file.
    Returns dictionary mapping document IDs to document data.
    """
   
def load_queries(file_path: str) -> Dict[str, Dict]:
    """
    TODO

    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    

def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    TODO
    
    Load relevance judgments from TSV file.
    Returns dictionary mapping query IDs to candidate relevance scores.
    """