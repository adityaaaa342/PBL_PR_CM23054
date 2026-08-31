"""
=============================================================================
College Project: Community Detection in Social Networks
Course: Pattern Recognition
Student: Aditya Mahalle (USN: CM23054)

What this project does:
1. Loads real-world social networks (College Football, Facebook Circles, DBLP).
2. Cleans the network (removes empty nodes and loops).
3. Finds friend groups using 3 standard algorithms (Louvain, Girvan-Newman, LPA).
4. Calculates accuracy (Modularity Q and NMI score).
5. Draws colorful pictures of the friend groups and saves them.
=============================================================================
"""

import os
import time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from datasets import get_football_network, get_facebook_circles, get_coauthorship_network, load_custom_csv

# =============================================================================
# STEP 1 & 2: LOAD DATA AND CLEAN THE GRAPH
# =============================================================================

def load_and_clean_graph(dataset_name="football", filepath=None):
    """
    Step 1 & 2: Load the social network data and clean it.
    - Remove self-loops (a person connected to themselves)
    - Remove isolated nodes (people with zero friends)
    """
    if filepath and os.path.exists(filepath):
        G, info = load_custom_csv(filepath)
    elif dataset_name == "football":
        G, info = get_football_network()
    elif dataset_name == "facebook":
        G, info = get_facebook_circles()
    elif dataset_name == "coauthorship":
        G, info = get_coauthorship_network()
    else:
        G, info = get_football_network()

    # Clean the graph:
    # 1. Remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    # 2. Make sure it is undirected
    if G.is_directed():
        G = G.to_undirected()
    # 3. Remove nodes with no connections
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G, info


# =============================================================================
# STEP 3: RUN THE 3 COMMUNITY DETECTION ALGORITHMS
# =============================================================================

def run_louvain(G):
    """
    Algorithm 1: Louvain Method
    - How it works: Moves nodes into groups to maximize connection density (Modularity).
    - Speed: Very fast (runs in milliseconds).
    """
    start_time = time.time()
    
    # Find communities
    comm_sets = list(nx.community.louvain_communities(G, seed=42))
    exec_time_ms = (time.time() - start_time) * 1000
    
    # Map each person to their group ID (0, 1, 2...)
    partition = {node: c_id for c_id, members in enumerate(comm_sets) for node in members}
    communities = [sorted(list(c)) for c in comm_sets]
    
    return {
        "algorithm": "Louvain",
        "communities": communities,
        "partition": partition,
        "num_communities": len(communities),
        "time_ms": exec_time_ms
    }

def run_girvan_newman(G, target_k=None):
    """
    Algorithm 2: Girvan-Newman Method
    - How it works: Finds and cuts the 'bridge' links between different friend circles.
    - Speed: Slower, because it recalculates all paths after each cut.
    """
    start_time = time.time()
    comp_gen = nx.community.girvan_newman(G)
    
    best_comms = None
    best_mod = -1.0
    
    # Progressively cut edges
    for step, comm_tuple in enumerate(comp_gen):
        comm_list = [sorted(list(c)) for c in comm_tuple]
        
        # Stop if we reach the target number of groups
        if target_k and len(comm_list) >= target_k:
            best_comms = comm_list
            break
            
        current_mod = nx.community.modularity(G, [set(c) for c in comm_list])
        if current_mod > best_mod:
            best_mod = current_mod
            best_comms = comm_list
            
        if step >= 15:  # stop after 15 splits for fast execution
            break
            
    if best_comms is None:
        best_comms = [sorted(list(G.nodes()))]
        
    partition = {node: c_id for c_id, members in enumerate(best_comms) for node in members}
    exec_time_ms = (time.time() - start_time) * 1000
    
    return {
        "algorithm": "Girvan-Newman",
        "communities": best_comms,
        "partition": partition,
        "num_communities": len(best_comms),
        "time_ms": exec_time_ms
    }

def run_label_propagation(G):
    """
    Algorithm 3: Label Propagation (LPA)
    - How it works: Each person starts with a color and adopts whatever color most friends have.
    - Speed: Super fast (nearly instant).
    """
    start_time = time.time()
    
    comm_sets = list(nx.community.asyn_lpa_communities(G, seed=42))
    exec_time_ms = (time.time() - start_time) * 1000
    
    partition = {node: c_id for c_id, members in enumerate(comm_sets) for node in members}
    communities = [sorted(list(c)) for c in comm_sets]
    
    return {
        "algorithm": "Label Propagation",
        "communities": communities,
        "partition": partition,
        "num_communities": len(communities),
        "time_ms": exec_time_ms
    }


# =============================================================================
# STEP 4: EVALUATE RESULTS (HOW GOOD ARE THE GROUPS?)
# =============================================================================

def evaluate_communities(G, result, ground_truth=None):
    """
    Step 4: Check quality using standard metrics.
    1. Modularity Q: Measures density inside groups (higher than 0.3 is great).
    2. Conductance: Measures cuts going outside (lower is better).
    3. NMI Score: Compares found groups with actual true groups (0 to 1 scale).
    """
    communities = result["communities"]
    partition = result["partition"]
    
    # 1. Calculate Modularity Q
    modularity = float(nx.community.modularity(G, [set(c) for c in communities]))
    
    # 2. Calculate Conductance
    conductances = []
    for c_nodes in communities:
        if 0 < len(c_nodes) < G.number_of_nodes():
            try:
                cond = nx.conductance(G, set(c_nodes))
                conductances.append(cond)
            except Exception:
                pass
    avg_conductance = float(np.mean(conductances)) if conductances else 0.0
    
    # 3. Calculate NMI and ARI (if ground truth labels are available)
    nmi, ari = None, None
    if ground_truth:
        common_nodes = sorted(list(set(partition.keys()) & set(ground_truth.keys())))
        y_pred = [partition[n] for n in common_nodes]
        y_true = [ground_truth[n] for n in common_nodes]
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))
        
    return {
        "algorithm": result["algorithm"],
        "num_communities": result["num_communities"],
        "modularity": round(modularity, 4),
        "conductance": round(avg_conductance, 4),
        "nmi": round(nmi, 4) if nmi is not None else "N/A",
        "ari": round(ari, 4) if ari is not None else "N/A",
        "time_ms": round(result["time_ms"], 2)
    }


# =============================================================================
# STEP 5: DRAW PICTURES AND SAVE EXPORTS
# =============================================================================

def plot_community_map(G, partition, title="Community Map", save_path="community_map.png"):
    """
    Step 5: Draw a 2D network diagram with distinct colors for each group.
    """
    plt.figure(figsize=(10, 8))
    
    # Color palette
    colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", 
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", 
        "#86BCB6", "#D37295"
    ]
    node_colors = [colors[partition.get(n, 0) % len(colors)] for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)
    
    # Draw graph elements
    nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="gray", width=1.0)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=350, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")
    
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[*] Saved Image: {save_path}")

def export_to_gephi(G, partition, filename="network_communities.gexf"):
    """
    Export the graph to Gephi (.gexf) so we can open it in 3D graph software.
    """
    G_export = G.copy()
    for n in G_export.nodes():
        G_export.nodes[n]["community"] = int(partition.get(n, 0))
    nx.write_gexf(G_export, filename)
    print(f"[*] Exported Gephi File: {filename}")


# =============================================================================
# MAIN FUNCTION: RUNS ALL BENCHMARKS
# =============================================================================

def benchmark_all_datasets():
    datasets = ["football", "facebook", "coauthorship"]
    
    print("=" * 80)
    print("  COMMUNITY DETECTION IN SOCIAL NETWORKS: EXPERIMENTAL BENCHMARK")
    print("=" * 80)
    
    for ds_key in datasets:
        G, info = load_and_clean_graph(ds_key)
        gt = info.get("ground_truth")
        
        print(f"\n>>> Running on: {info['name']} ({G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges)")
        
        # Step 3: Run the 3 algorithms
        res_louvain = run_louvain(G)
        res_gn = run_girvan_newman(G, target_k=info.get('ground_truth_k'))
        res_lpa = run_label_propagation(G)
        
        # Step 4: Evaluate
        e_louvain = evaluate_communities(G, res_louvain, gt)
        e_gn = evaluate_communities(G, res_gn, gt)
        e_lpa = evaluate_communities(G, res_lpa, gt)
        
        # Display table
        df = pd.DataFrame([e_louvain, e_gn, e_lpa])
        print(df.to_string(index=False))
        
        # Step 5: Save image outputs
        plot_community_map(G, res_louvain["partition"], f"{info['name']} - Louvain (Q={e_louvain['modularity']})", f"{ds_key}_louvain.png")
        plot_community_map(G, res_gn["partition"], f"{info['name']} - Girvan-Newman (Q={e_gn['modularity']})", f"{ds_key}_gn.png")
        plot_community_map(G, res_lpa["partition"], f"{info['name']} - Label Propagation (Q={e_lpa['modularity']})", f"{ds_key}_lpa.png")
        export_to_gephi(G, res_louvain["partition"], f"{ds_key}_communities.gexf")
        
    print("\n" + "=" * 80)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    benchmark_all_datasets()
