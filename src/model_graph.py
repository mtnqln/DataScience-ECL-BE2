from typing import Dict
from handle_data import load_corpus
import networkx as nx
from networkx import DiGraph
from utils import use_cache


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
    # print(f"Betweenness centrality : {bet}") 
    print(f"Page rank : {pr}")

    
    
