from typing import Dict
from .handle_data import load_corpus
import networkx as nx
from networkx import DiGraph
from .utils import use_cache
import numpy as np

def build_graph(corpus:Dict[str,Dict])->DiGraph:
    G = DiGraph()
    G.add_nodes_from(corpus.keys())
    for (key,value) in corpus.items():
        cited_by = value["metadata"]["cited_by"]
        references = value["metadata"]["references"]
        # A -> B means B is cited by A
        for c in cited_by:
            G.add_edge(c,key) 
        for r in references:
            G.add_edge(key,r)
    return G

@use_cache
def calculate_elemtary_indicator(g:DiGraph):
    node_number = len(g.nodes)
    edge_number = len(g.edges)
    variance = lambda x,x_mean : (1/len(x)) * sum((x_i-x_mean)**2 for x_i in x)

    graph_density = edge_number / (node_number *(node_number-1))
    graph_degree = edge_number/node_number
    graph_in_degree_variance = variance([g.in_degree(node) for node in g.nodes],edge_number/node_number)
    graph_out_degree_variance = variance([g.out_degree(node) for node in g.nodes],edge_number/node_number)

    return node_number,edge_number,graph_density,graph_degree,graph_in_degree_variance,graph_out_degree_variance

@use_cache
def calculate_centrality_indicator(g:DiGraph):
    deg = nx.degree_centrality(g)
    bet = nx.betweenness_centrality(g,k=100)
    pr = nx.pagerank(g)

    return deg,bet,pr

def improve_embedding(corpus_embeddings, G: nx.DiGraph, id_to_index, alpha=0.3, beta=0.2):
    """
    Improve document embeddings using citation graph structure.
    
    Combines the original embedding with information from:
    - Papers that this document cites (successors/references)
    - Papers that cite this document (predecessors/cited_by)
    
    Args:
        corpus_embeddings: Original document embeddings (numpy array)
        G: Citation graph (NetworkX DiGraph)
        id_to_index: Mapping from document ID to embedding index
        alpha: Weight for cited papers (references), default 0.3
        beta: Weight for citing papers (cited_by), default 0.2
        
    Returns:
        Enhanced embeddings incorporating graph structure
    """
    graph_embeddings = np.copy(corpus_embeddings)
    
    print(f"Calcul des représentations graphiques (alpha={alpha}, beta={beta})...")
    print(f"  - alpha: poids des articles cités (references)")
    print(f"  - beta: poids des articles citants (cited_by)")
    
    nodes_improved = 0
    nodes_with_refs = 0
    nodes_with_citations = 0

    for doc_id in G.nodes():
        if doc_id not in id_to_index:
            continue
            
        current_idx = id_to_index[doc_id]
        
        # Get papers cited by this document (successors in the graph)
        cited_papers = list(G.successors(doc_id))
        valid_cited_indices = [id_to_index[n] for n in cited_papers if n in id_to_index]
        
        # Get papers that cite this document (predecessors in the graph)
        citing_papers = list(G.predecessors(doc_id))
        valid_citing_indices = [id_to_index[n] for n in citing_papers if n in id_to_index]
        
        # Start with original embedding
        new_vec = corpus_embeddings[current_idx].copy()
        total_weight = 1.0
        
        # Add information from cited papers (references)
        if len(valid_cited_indices) > 0:
            cited_vectors = corpus_embeddings[valid_cited_indices]
            mean_cited_vec = np.mean(cited_vectors, axis=0)
            new_vec = new_vec + alpha * mean_cited_vec
            total_weight += alpha
            nodes_with_refs += 1
        
        # Add information from citing papers (cited_by)
        if len(valid_citing_indices) > 0:
            citing_vectors = corpus_embeddings[valid_citing_indices]
            mean_citing_vec = np.mean(citing_vectors, axis=0)
            new_vec = new_vec + beta * mean_citing_vec
            total_weight += beta
            nodes_with_citations += 1
        
        # Normalize to maintain embedding scale
        if total_weight > 1.0:
            new_vec = new_vec / total_weight
            graph_embeddings[current_idx] = new_vec
            nodes_improved += 1

    print(f"Nouveaux embeddings calculés !")
    print(f"  - Nœuds améliorés: {nodes_improved}/{len(G.nodes())}")
    print(f"  - Nœuds avec références: {nodes_with_refs}")
    print(f"  - Nœuds avec citations: {nodes_with_citations}")
    
    return graph_embeddings


def improve_embedding_advanced(corpus_embeddings, G: nx.DiGraph, id_to_index, 
                               alpha=0.4, beta=0.3, use_pagerank=True, normalize_l2=True):
    """
    Advanced version with L2 normalization and PageRank weighting.
    
    Args:
        corpus_embeddings: Original document embeddings (numpy array)
        G: Citation graph (NetworkX DiGraph)
        id_to_index: Mapping from document ID to embedding index
        alpha: Weight for cited papers (references), default 0.4
        beta: Weight for citing papers (cited_by), default 0.3
        use_pagerank: If True, weight neighbors by their PageRank score
        normalize_l2: If True, apply L2 normalization to final embeddings
        
    Returns:
        Enhanced embeddings with advanced optimizations
    """
    from sklearn.preprocessing import normalize
    
    graph_embeddings = np.copy(corpus_embeddings)
    
    print(f"Calcul des représentations graphiques AVANCÉES...")
    print(f"  - alpha={alpha}, beta={beta}")
    print(f"  - PageRank weighting: {use_pagerank}")
    print(f"  - L2 normalization: {normalize_l2}")
    
    # Calculate PageRank if needed
    pagerank_scores = None
    if use_pagerank:
        print("  - Calcul du PageRank...")
        pagerank_scores = nx.pagerank(G, alpha=0.85)
    
    nodes_improved = 0
    nodes_with_refs = 0
    nodes_with_citations = 0

    for doc_id in G.nodes():
        if doc_id not in id_to_index:
            continue
            
        current_idx = id_to_index[doc_id]
        
        # Get papers cited by this document
        cited_papers = list(G.successors(doc_id))
        valid_cited_indices = [id_to_index[n] for n in cited_papers if n in id_to_index]
        valid_cited_ids = [n for n in cited_papers if n in id_to_index]
        
        # Get papers that cite this document
        citing_papers = list(G.predecessors(doc_id))
        valid_citing_indices = [id_to_index[n] for n in citing_papers if n in id_to_index]
        valid_citing_ids = [n for n in citing_papers if n in id_to_index]
        
        # Start with original embedding
        new_vec = corpus_embeddings[current_idx].copy()
        total_weight = 1.0
        
        # Add information from cited papers with optional PageRank weighting
        if len(valid_cited_indices) > 0:
            cited_vectors = corpus_embeddings[valid_cited_indices]
            
            if use_pagerank and pagerank_scores:
                # Weighted average by PageRank
                weights = np.array([pagerank_scores.get(doc_id, 1.0) for doc_id in valid_cited_ids])
                weights = weights / weights.sum()  # Normalize weights
                mean_cited_vec = np.average(cited_vectors, axis=0, weights=weights)
            else:
                mean_cited_vec = np.mean(cited_vectors, axis=0)
            
            new_vec = new_vec + alpha * mean_cited_vec
            total_weight += alpha
            nodes_with_refs += 1
        
        # Add information from citing papers with optional PageRank weighting
        if len(valid_citing_indices) > 0:
            citing_vectors = corpus_embeddings[valid_citing_indices]
            
            if use_pagerank and pagerank_scores:
                # Weighted average by PageRank
                weights = np.array([pagerank_scores.get(doc_id, 1.0) for doc_id in valid_citing_ids])
                weights = weights / weights.sum()  # Normalize weights
                mean_citing_vec = np.average(citing_vectors, axis=0, weights=weights)
            else:
                mean_citing_vec = np.mean(citing_vectors, axis=0)
            
            new_vec = new_vec + beta * mean_citing_vec
            total_weight += beta
            nodes_with_citations += 1
        
        # Normalize to maintain embedding scale
        if total_weight > 1.0:
            new_vec = new_vec / total_weight
            graph_embeddings[current_idx] = new_vec
            nodes_improved += 1
    
    # Apply L2 normalization to all embeddings
    if normalize_l2:
        print("  - Application de la normalisation L2...")
        graph_embeddings = normalize(graph_embeddings, norm='l2', axis=1)

    print(f"Nouveaux embeddings calculés !")
    print(f"  - Nœuds améliorés: {nodes_improved}/{len(G.nodes())}")
    print(f"  - Nœuds avec références: {nodes_with_refs}")
    print(f"  - Nœuds avec citations: {nodes_with_citations}")
    
    return graph_embeddings



if __name__=="__main__":
    corpus: Dict[str, Dict] = load_corpus("data/corpus.jsonl")
    g = build_graph(corpus)


    ### Indicateur elementaires
    node_number,edge_number,graph_density,graph_degree,graph_in_degree_variance,graph_out_degree_variance = calculate_elemtary_indicator(g)
    print(f"Nodes number : {node_number}")
    print(f"Edges number : {edge_number}")
    print(f"Graph density : {graph_density }")
    print(f"Graph degree mean : {graph_degree}")
    print(f"Graph In degree variance : {graph_in_degree_variance}")
    print(f"Graph Out degree variance : {graph_out_degree_variance}")

    ### Indicateurs de centralites
    deg,bet,pr = calculate_centrality_indicator(g)
    print(f"Centralite deg : {deg}")
    print(f"Page rank : {pr}")

    
    
