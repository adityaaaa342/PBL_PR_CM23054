"""
=============================================================================
Course: Pattern Recognition
Project: Community Detection in Social Networks
Author: Aditya Mahalle (USN: CM23054)
Backend: Flask Application for Render & Vercel Deployment
=============================================================================
"""

import os
import io
import time
import json
import networkx as nx
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from datasets import get_football_network, get_facebook_circles, get_coauthorship_network
from algorithms import detect_louvain, detect_girvan_newman, detect_label_propagation, detect_spectral

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload


# Cache active datasets in memory
GRAPH_CACHE = {}

def get_graph_by_id(dataset_id):
    if dataset_id in GRAPH_CACHE:
        return GRAPH_CACHE[dataset_id]
    
    if dataset_id == "football":
        G, info = get_football_network()
    elif dataset_id == "facebook":
        G, info = get_facebook_circles()
    elif dataset_id == "coauthorship":
        G, info = get_coauthorship_network()
    else:
        G, info = get_football_network()
        
    # Clean Graph
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.is_directed():
        G = G.to_undirected()
    G.remove_nodes_from(list(nx.isolates(G)))
    
    GRAPH_CACHE[dataset_id] = (G, info)
    return G, info


@app.route("/")
@app.route("/api/index")
@app.route("/api")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def serve_static_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    datasets = [
        {
            "id": "football",
            "name": "American College Football Network",
            "reference": "Girvan & Newman (2002)",
            "nodes": 115,
            "edges": 613,
            "ground_truth_k": 12,
            "category": "Sports / Athletic Conferences",
            "description": "115 US college teams playing regular season games partitioned across 12 athletic conferences."
        },
        {
            "id": "facebook",
            "name": "Facebook Social Circles (SNAP)",
            "reference": "McAuley & Leskovec (NeurIPS 2012)",
            "nodes": 78,
            "edges": 413,
            "ground_truth_k": 4,
            "category": "Social Media Circles",
            "description": "78 users interconnected across 4 distinct circles: University, School, Work, Hobbies."
        },
        {
            "id": "coauthorship",
            "name": "Academic Co-Authorship Network (DBLP)",
            "reference": "SNAP DBLP Collaboration Network",
            "nodes": 120,
            "edges": 995,
            "ground_truth_k": 3,
            "category": "Academic Citations",
            "description": "120 researchers collaborating across Machine Learning, Databases, and Computer Networks."
        }
    ]
    return jsonify({"success": True, "datasets": datasets})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.get_json() or {}
    dataset_id = data.get("dataset", "football")
    algorithm = data.get("algorithm", "louvain")
    target_k = data.get("target_k", None)
    resolution = float(data.get("resolution", 1.0))
    
    G, info = get_graph_by_id(dataset_id)
    ground_truth = info.get("ground_truth")
    
    # Run chosen algorithm
    start_time = time.time()
    if algorithm == "louvain":
        result = detect_louvain(G, resolution=resolution)
    elif algorithm == "girvan_newman":
        k_val = int(target_k) if target_k else info.get("ground_truth_k", 4)
        result = detect_girvan_newman(G, num_clusters=k_val)
    elif algorithm == "lpa":
        result = detect_label_propagation(G)
    elif algorithm == "spectral":
        k_val = int(target_k) if target_k else info.get("ground_truth_k", 3)
        result = detect_spectral(G, k=k_val)
    else:
        result = detect_louvain(G)
    
    exec_time_ms = result.get("time_ms", (time.time() - start_time) * 1000)
    partition = result["partition"]
    communities = result["communities"]
    
    # Compute Metrics
    modularity = float(nx.community.modularity(G, [set(c) for c in communities]))
    
    conductances = []
    for c_nodes in communities:
        if 0 < len(c_nodes) < G.number_of_nodes():
            try:
                cond = nx.conductance(G, set(c_nodes))
                conductances.append(cond)
            except Exception:
                pass
    avg_conductance = float(np.mean(conductances)) if conductances else 0.0
    
    nmi, ari = None, None
    if ground_truth:
        common_nodes = sorted(list(set(partition.keys()) & set(ground_truth.keys())))
        y_pred = [partition[n] for n in common_nodes]
        y_true = [ground_truth[n] for n in common_nodes]
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))
        
    # Prepare Vis.js graph nodes and edges
    # Community color palette
    palette = [
        "#00F0FF", "#FF007F", "#7000FF", "#00FF66", "#FFE600",
        "#FF6B00", "#9D00FF", "#00E5FF", "#FF2A85", "#00FFA3",
        "#FFB800", "#7B61FF", "#FF453A", "#30D158", "#BF5AF2"
    ]
    
    # Generate spring layout for smooth initial coordinates
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    node_list = []
    for n in G.nodes():
        cid = partition.get(n, 0)
        color = palette[cid % len(palette)]
        deg = G.degree[n]
        label_text = str(G.nodes[n].get("label", str(n)))
        extra_attr = G.nodes[n].get("conference") or G.nodes[n].get("domain") or ""
        
        node_list.append({
            "id": str(n),
            "label": label_text,
            "title": f"<b>Node:</b> {label_text}<br><b>Community:</b> {cid + 1}<br><b>Degree:</b> {deg}" + (f"<br><b>Faction/Conf:</b> {extra_attr}" if extra_attr else ""),
            "group": int(cid),
            "color": {
                "background": color,
                "border": "#FFFFFF",
                "highlight": {"background": "#FFFFFF", "border": color},
                "hover": {"background": color, "border": "#FFFFFF"}
            },
            "size": 12 + min(deg * 1.5, 20),
            "x": float(pos[n][0] * 800),
            "y": float(pos[n][1] * 800)
        })
        
    edge_list = []
    for u, v in G.edges():
        same_comm = partition.get(u) == partition.get(v)
        edge_color = palette[partition.get(u, 0) % len(palette)] if same_comm else "#4A5568"
        edge_list.append({
            "from": str(u),
            "to": str(v),
            "color": {"color": edge_color, "opacity": 0.45 if same_comm else 0.15},
            "width": 1.5 if same_comm else 0.8
        })
        
    # Community breakdown distribution
    community_sizes = [{"id": cid + 1, "size": len(c), "color": palette[cid % len(palette)]} for cid, c in enumerate(communities)]
    community_sizes = sorted(community_sizes, key=lambda x: x["size"], reverse=True)
    
    return jsonify({
        "success": True,
        "dataset_name": info["name"],
        "algorithm": result.get("name", algorithm.capitalize()),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_communities": len(communities),
        "metrics": {
            "modularity": round(modularity, 4),
            "conductance": round(avg_conductance, 4),
            "nmi": round(nmi, 4) if nmi is not None else None,
            "ari": round(ari, 4) if ari is not None else None,
            "time_ms": round(exec_time_ms, 2)
        },
        "community_breakdown": community_sizes,
        "graph_data": {
            "nodes": node_list,
            "edges": edge_list
        }
    })


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = request.get_json() or {}
    dataset_id = data.get("dataset", "football")
    
    G, info = get_graph_by_id(dataset_id)
    gt = info.get("ground_truth")
    k_val = info.get("ground_truth_k", 4)
    
    # 1. Louvain
    res_louvain = detect_louvain(G)
    
    # 2. Girvan-Newman
    res_gn = detect_girvan_newman(G, num_clusters=k_val)
    
    # 3. LPA
    res_lpa = detect_label_propagation(G)
    
    # 4. Spectral
    res_spectral = detect_spectral(G, k=k_val if k_val else 3)
    
    algos = [
        ("Louvain", res_louvain),
        ("Girvan-Newman", res_gn),
        ("Label Propagation", res_lpa),
        ("Spectral Clustering", res_spectral)
    ]
    
    results = []
    for name, res in algos:
        comms = res["communities"]
        part = res["partition"]
        q = float(nx.community.modularity(G, [set(c) for c in comms]))
        
        conds = [nx.conductance(G, set(c)) for c in comms if 0 < len(c) < G.number_of_nodes()]
        avg_cond = float(np.mean(conds)) if conds else 0.0
        
        nmi, ari = None, None
        if gt:
            common = sorted(list(set(part.keys()) & set(gt.keys())))
            y_pred = [part[n] for n in common]
            y_true = [gt[n] for n in common]
            nmi = float(normalized_mutual_info_score(y_true, y_pred))
            ari = float(adjusted_rand_score(y_true, y_pred))
            
        results.append({
            "algorithm": name,
            "communities": len(comms),
            "modularity": round(q, 4),
            "conductance": round(avg_cond, 4),
            "nmi": round(nmi, 4) if nmi is not None else None,
            "ari": round(ari, 4) if ari is not None else None,
            "time_ms": round(res["time_ms"], 2)
        })
        
    return jsonify({
        "success": True,
        "dataset_name": info["name"],
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "benchmark": results
    })


@app.route("/api/upload_csv", methods=["POST"])
def api_upload_csv():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400
        
    try:
        df = pd.read_csv(file)
        if len(df.columns) < 2:
            return jsonify({"success": False, "error": "CSV must have at least 2 columns (Source, Target)"}), 400
            
        src_col = df.columns[0]
        dst_col = df.columns[1]
        
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(str(row[src_col]), str(row[dst_col]))
            
        # Clean graph
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        
        custom_id = f"custom_{int(time.time())}"
        info = {
            "id": custom_id,
            "name": f"Custom Graph ({file.filename})",
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "ground_truth_k": None,
            "ground_truth": None,
            "description": f"Uploaded custom edge list from {file.filename} with {G.number_of_nodes()} nodes."
        }
        
        GRAPH_CACHE[custom_id] = (G, info)
        
        return jsonify({
            "success": True,
            "dataset_id": custom_id,
            "name": info["name"],
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Community Detection Web Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
