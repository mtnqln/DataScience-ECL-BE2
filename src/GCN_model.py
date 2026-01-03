import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(SimpleGCN, self).__init__()
        # Pour une propagation simple (SGC), nous n'avons pas forcément besoin de couches linéaires entraînables
        # si nous voulons juste lisser.
        # Mais pour un GCN complet, on peut ajouter une transformation linéaire.
        # Ici, nous implémentons une version "light" qui permet la propagation multi-sauts
        # qui est souvent la clé pour ce type de tâche (smoothing).
        
        self.k_hops = 2 # Nombre d'itérations de propagation
        
    def get_adjacency_matrix(self, G, node_order):
        """
        Construit la matrice d'adjacence normalisée pour GCN: D^-1/2 (A + I) D^-1/2
        On rend le graphe symétrique pour une meilleure propagation.
        """
        print("Construction de la matrice d'adjacence...")
        
        # Création d'un mapping pour s'assurer de l'ordre
        id_to_idx = {doc_id: i for i, doc_id in enumerate(node_order)}
        num_nodes = len(node_order)
        
        # Construction des arêtes (symétrique)
        edges = []
        for u, v in G.edges():
            if u in id_to_idx and v in id_to_idx:
                idx_u = id_to_idx[u]
                idx_v = id_to_idx[v]
                edges.append((idx_u, idx_v))
                edges.append((idx_v, idx_u)) # Symétrisation
        
        # Ajout des self-loops
        for i in range(num_nodes):
            edges.append((i, i))
            
        # Conversion en sparse matrix pour PyTorch
        if not edges:
            # Fallback si pas d'arêtes (ne devrait pas arriver)
            return torch.eye(num_nodes).to_sparse()

        edges = np.array(edges).T
        values = np.ones(edges.shape[1])
        
        adj_coo = coo_matrix((values, (edges[0], edges[1])), shape=(num_nodes, num_nodes))
        
        # Normalisation D^-1/2 A D^-1/2
        rowsum = np.array(adj_coo.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        d_mat_inv_sqrt = coo_matrix((d_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes))
        
        norm_adj = d_mat_inv_sqrt.dot(adj_coo).dot(d_mat_inv_sqrt)
        
        # Conversion en Tensor PyTorch Sparse
        norm_adj_coo = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((norm_adj_coo.row, norm_adj_coo.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj_coo.data.astype(np.float32))
        shape = torch.Size(norm_adj_coo.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self, features, adj, k=2):
        """
        Propulse les features à travers le graphe k fois.
        X' = (A_norm)^k * X
        """
        x = features
        for i in range(k):
            # Sparse Matrix Multiplication: A * X
            # torch.sparse.mm ne supporte que sparse * dense
            x = torch.sparse.mm(adj, x)
            # On pourrait ajouter une non-linéarité ici si on avait des poids (ReLU, etc.)
            # Mais pour du SGC (Simple Graph Convolution), la linearité est souvent optimale pour le lissage
        return x
