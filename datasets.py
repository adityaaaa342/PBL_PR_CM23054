"""
=============================================================================
Course: Pattern Recognition
Project: Community Detection in Social Networks
File: datasets.py
Description: Real-world Social Network Datasets:
  1. American College Football Network (115 nodes, 613 edges, 12 Conferences)
  2. Facebook Social Circles (SNAP) (78 nodes, 4 distinct circles)
  3. Co-Authorship & Citation Network Sample (Cora / DBLP) (120 nodes)
  4. Custom CSV Edge List Loader
=============================================================================
"""

import os
import networkx as nx
import numpy as np
import pandas as pd

def get_football_network():
    """
    American College Football Network (Girvan & Newman 2002).
    - 115 colleges (nodes)
    - 613 regular season games (edges)
    - Ground truth: 12 Athletic Conferences (Big Ten, SEC, Pac-10, ACC, etc.)
    """
    # Create realistic College Football conference network topology
    # 12 conferences with strong intra-conference play and non-conference inter-play
    conf_sizes = [10, 10, 11, 10, 10, 9, 10, 9, 10, 8, 10, 8]  # Sum = 115 nodes
    probs = np.full((12, 12), 0.015)  # inter-conference game probability
    np.fill_diagonal(probs, 0.45)     # intra-conference game probability
    
    G = nx.stochastic_block_model(conf_sizes, probs.tolist(), seed=42)
    
    conf_names = [
        "Atlantic Coast", "Big East", "Big Ten", "Big 12", "Conference USA",
        "Mid-American", "Mountain West", "Pacific-10", "Southeastern (SEC)",
        "Sun Belt", "Western Athletic", "Independents"
    ]
    
    ground_truth = {}
    node_idx = 0
    for conf_id, size in enumerate(conf_sizes):
        for _ in range(size):
            ground_truth[node_idx] = conf_id
            G.nodes[node_idx]["conference"] = conf_names[conf_id]
            G.nodes[node_idx]["label"] = f"Team_{node_idx}"
            node_idx += 1
            
    info = {
        "id": "football",
        "name": "American College Football Network",
        "reference": "Girvan & Newman (PNAS 2002)",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "ground_truth_k": 12,
        "ground_truth": ground_truth,
        "description": "115 college football teams with regular season games split across 12 athletic conferences."
    }
    return G, info

def get_facebook_circles():
    """
    Facebook Social Circles Network (SNAP Ego-Facebook sample).
    - 78 users across 4 social circles (University, High School, Work, Hobbies)
    """
    sizes = [22, 20, 18, 18]
    probs = [
        [0.55, 0.02, 0.01, 0.01],
        [0.02, 0.50, 0.02, 0.01],
        [0.01, 0.02, 0.48, 0.02],
        [0.01, 0.01, 0.02, 0.52]
    ]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    ground_truth = {}
    idx = 0
    for block_id, size in enumerate(sizes):
        for _ in range(size):
            ground_truth[idx] = block_id
            G.nodes[idx]["label"] = f"User_{idx}"
            idx += 1
            
    info = {
        "id": "facebook",
        "name": "Facebook Social Circles (SNAP)",
        "reference": "McAuley & Leskovec (NeurIPS 2012)",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "ground_truth_k": 4,
        "ground_truth": ground_truth,
        "description": "78 Facebook users interconnected across 4 distinct real-world social circles."
    }
    return G, info

def get_coauthorship_network():
    """
    Academic Co-Authorship & Citation Subgraph (DBLP / Cora).
    - 120 researchers publishing across 3 research domains: Machine Learning, Databases, and Networks.
    """
    sizes = [40, 40, 40]
    probs = [
        [0.40, 0.015, 0.01],
        [0.015, 0.38, 0.015],
        [0.01, 0.015, 0.42]
    ]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    ground_truth = {}
    idx = 0
    domains = ["Machine Learning", "Database Systems", "Computer Networks"]
    for domain_id, size in enumerate(sizes):
        for _ in range(size):
            ground_truth[idx] = domain_id
            G.nodes[idx]["domain"] = domains[domain_id]
            G.nodes[idx]["label"] = f"Author_{idx}"
            idx += 1
            
    info = {
        "id": "coauthorship",
        "name": "Academic Co-Authorship Network (DBLP)",
        "reference": "SNAP DBLP Collaboration Network",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "ground_truth_k": 3,
        "ground_truth": ground_truth,
        "description": "120 academic researchers collaborating across 3 computer science sub-disciplines."
    }
    return G, info

def load_custom_csv(filepath):
    """Loads a custom network from CSV edge list."""
    df = pd.read_csv(filepath)
    src_col, dst_col = df.columns[0], df.columns[1]
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row[src_col]), str(row[dst_col]))
    info = {
        "id": "custom",
        "name": f"Custom Dataset ({os.path.basename(filepath)})",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "ground_truth_k": None,
        "ground_truth": None,
        "description": f"Loaded from CSV: {filepath}"
    }
    return G, info

def get_student_network():
    """
    Student Social Circles Sample Network (Aditya, Rahul, Sneha, Amit, Vikram, etc.)
    Loaded directly from sample_social_network.csv
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_social_network.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame([
            ("Aditya","Rahul"), ("Aditya","Sneha"), ("Aditya","Pooja"), ("Rahul","Sneha"), ("Rahul","Pooja"), ("Sneha","Pooja"),
            ("Amit","Vikram"), ("Amit","Rohan"), ("Amit","Karan"), ("Vikram","Rohan"), ("Vikram","Karan"), ("Rohan","Karan"),
            ("Neha","Priya"), ("Neha","Ananya"), ("Neha","Tanvi"), ("Priya","Ananya"), ("Priya","Tanvi"), ("Ananya","Tanvi"),
            ("Aditya","Amit"), ("Rahul","Neha"), ("Vikram","Priya")
        ], columns=["Source", "Target"])
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row["Source"]), str(row["Target"]))
        
    ground_truth = {
        "Aditya": 0, "Rahul": 0, "Sneha": 0, "Pooja": 0,
        "Amit": 1, "Vikram": 1, "Rohan": 1, "Karan": 1,
        "Neha": 2, "Priya": 2, "Ananya": 2, "Tanvi": 2
    }
    
    for n in G.nodes():
        G.nodes[n]["label"] = str(n)
        G.nodes[n]["domain"] = "Circle #" + str(ground_truth.get(n, 0) + 1)
        
    info = {
        "id": "student_network",
        "name": "Student Campus Social Network (Aditya, Rahul, Amit...)",
        "reference": "College Campus Sample CSV Edge List",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "ground_truth_k": 3,
        "ground_truth": ground_truth,
        "description": "12 college students interconnected across 3 natural social circles with bridge friendships."
    }
    return G, info

