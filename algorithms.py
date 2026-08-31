"""
=============================================================================
College Project: Community Detection in Social Networks
File: algorithms.py
Description: Implementation of standard community detection algorithms:
             1. Louvain Algorithm (Modularity Optimization)
             2. Girvan-Newman Algorithm (Edge Betweenness)
             3. Label Propagation Algorithm (LPA)
             4. Spectral Clustering (Graph Laplacian)
=============================================================================
"""

import time
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# -------------------------------------------------------------------------
# 1. Louvain Algorithm
# -------------------------------------------------------------------------
def detect_louvain(G, resolution=1.0):
    """
    Louvain Method:
    Heuristic algorithm that optimizes Newman's Modularity.
    Fast and works well on both small and large social networks.
    """
    start = time.time()
    
    # Run Louvain
    community_sets = list(nx.community.louvain_communities(G, resolution=resolution, seed=42))
    
    # Create node -> community dictionary
    partition = {}
    for comm_id, members in enumerate(community_sets):
        for node in members:
            partition[node] = comm_id
            
    # Calculate Modularity Q
    modularity = nx.community.modularity(G, community_sets)
    exec_time = (time.time() - start) * 1000  # in milliseconds
    
    return {
        "name": "Louvain Modularity",
        "communities": [sorted(list(c)) for c in community_sets],
        "partition": partition,
        "num_communities": len(community_sets),
        "modularity": round(modularity, 4),
        "time_ms": round(exec_time, 2)
    }


# -------------------------------------------------------------------------
# 2. Girvan-Newman Algorithm
# -------------------------------------------------------------------------
def detect_girvan_newman(G, num_clusters=None):
    """
    Girvan-Newman Method:
    Divisive hierarchical algorithm. It repeatedly finds and removes the edge
    with the highest 'edge betweenness centrality' (bridge edges between groups).
    """
    start = time.time()
    
    comp_generator = nx.community.girvan_newman(G)
    
    best_communities = None
    best_modularity = -1.0
    
    # Iterate through divisions
    for step, communities in enumerate(comp_generator):
        comm_list = [sorted(list(c)) for c in communities]
        
        # If user specified a specific number of clusters (e.g. k=2)
        if num_clusters and len(comm_list) >= num_clusters:
            best_communities = comm_list
            best_modularity = nx.community.modularity(G, [set(c) for c in comm_list])
            break
            
        # Otherwise pick the step with highest modularity
        current_mod = nx.community.modularity(G, [set(c) for c in comm_list])
        if current_mod > best_modularity:
            best_modularity = current_mod
            best_communities = comm_list
            
        if step >= 10:  # limit search steps for faster execution
            break
            
    if best_communities is None:
        best_communities = [sorted(list(G.nodes()))]
        best_modularity = 0.0
        
    partition = {}
    for comm_id, members in enumerate(best_communities):
        for node in members:
            partition[node] = comm_id
            
    exec_time = (time.time() - start) * 1000
    
    return {
        "name": "Girvan-Newman",
        "communities": best_communities,
        "partition": partition,
        "num_communities": len(best_communities),
        "modularity": round(best_modularity, 4),
        "time_ms": round(exec_time, 2)
    }


# -------------------------------------------------------------------------
# 3. Label Propagation Algorithm (LPA)
# -------------------------------------------------------------------------
def detect_label_propagation(G):
    """
    Label Propagation Method:
    Each node starts with a unique label and iteratively adopts 
    the label that most of its neighbors have.
    Runs in near-linear time O(V + E).
    """
    start = time.time()
    
    community_sets = list(nx.community.asyn_lpa_communities(G, seed=42))
    
    partition = {}
    for comm_id, members in enumerate(community_sets):
        for node in members:
            partition[node] = comm_id
            
    modularity = nx.community.modularity(G, community_sets)
    exec_time = (time.time() - start) * 1000
    
    return {
        "name": "Label Propagation (LPA)",
        "communities": [sorted(list(c)) for c in community_sets],
        "partition": partition,
        "num_communities": len(community_sets),
        "modularity": round(modularity, 4),
        "time_ms": round(exec_time, 2)
    }


# -------------------------------------------------------------------------
# 4. Spectral Clustering
# -------------------------------------------------------------------------
def detect_spectral(G, k=2):
    """
    Spectral Clustering:
    Constructs the Normalized Graph Laplacian Matrix L = D^(-1/2) * (D - A) * D^(-1/2),
    finds the k smallest eigenvectors, and clusters nodes using K-Means.
    """
    start = time.time()
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n <= k:
        return detect_louvain(G)
        
    # Adjacency Matrix A and Degree vector
    A = nx.to_numpy_array(G, nodelist=nodes)
    degrees = np.sum(A, axis=1)
    
    # Normalized Laplacian L_sym = I - D^(-1/2) A D^(-1/2)
    d_inv_sqrt = np.zeros_like(degrees)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    
    # Take first k eigenvectors
    U = eigenvectors[:, :k]
    T = normalize(U, norm='l2', axis=1)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(T)
    
    partition = {nodes[i]: int(labels[i]) for i in range(n)}
    
    # Group nodes by cluster
    comm_dict = {}
    for node, c_id in partition.items():
        comm_dict.setdefault(c_id, []).append(node)
        
    community_list = [sorted(comm_dict[c]) for c in sorted(comm_dict.keys())]
    modularity = nx.community.modularity(G, [set(c) for c in community_list])
    exec_time = (time.time() - start) * 1000
    
    return {
        "name": "Spectral Clustering",
        "communities": community_list,
        "partition": partition,
        "num_communities": len(community_list),
        "modularity": round(modularity, 4),
        "time_ms": round(exec_time, 2)
    }
