# Community Detection in Social Networks

**Course:** Pattern Recognition  
**Student:** Aditya Mahalle (USN: CM23054)  

---

## 📌 Project Overview
This project implements an end-to-end **community detection pipeline** that discovers natural, tightly-knit groups (communities) inside social networks using graph topology and unsupervised algorithms.

The system features:
- **5 community detection algorithms** compared head-to-head
- **4 real-world benchmark datasets** with ground truth labels
- **Interactive web dashboard** with force-directed graph visualization
- **Comprehensive evaluation metrics**: Modularity, NMI, ARI, Conductance
- **Custom CSV upload** support for user-provided networks

---

## 🔬 Algorithms Implemented

| # | Algorithm | Method Category | Complexity | Description |
|---|-----------|----------------|------------|-------------|
| 1 | **Louvain** | Modularity Maximization | $\mathcal{O}(N \log N)$ | Fast greedy hierarchical agglomeration |
| 2 | **Girvan-Newman** | Edge Betweenness Divisive | $\mathcal{O}(M^2 N)$ | Iteratively removes highest-betweenness bridges |
| 3 | **Label Propagation (LPA)** | Dynamic Consensus | $\mathcal{O}(M + N)$ | Nodes adopt majority neighbor label |
| 4 | **Spectral Clustering** | Graph Laplacian | $\mathcal{O}(N^3)$ | Eigen-decomposition + K-Means clustering |
| 5 | **Leiden** | Improved Louvain | $\mathcal{O}(N \log N)$ | Guarantees well-connected communities |

---

## 📂 Datasets

| Dataset | Nodes | Edges | Ground Truth | Source |
|---------|-------|-------|-------------|--------|
| 🎓 Student Campus Network | 12 | 21 | 3 Friend Circles | Custom CSV |
| 🏈 American College Football | 115 | 613 | 12 Conferences | Girvan & Newman (2002) |
| 👥 Facebook Social Circles | 78 | 413 | 4 Social Circles | SNAP (McAuley & Leskovec) |
| 📚 Academic Co-Authorship | 120 | 995 | 3 Domains | DBLP Collaboration Network |

---

## 📊 Benchmark Results (American College Football Network)

| Algorithm | Communities | Modularity ($Q$) | Conductance | NMI | Runtime |
|---|---|---|---|---|---|
| **Louvain** | 10 | **0.6290** | 0.2689 | **91.95%** | 10.49 ms |
| **Girvan-Newman** | 12 | 0.6043 | 0.3242 | 88.66% | 1751.29 ms |
| **Label Propagation** | 14 | 0.5972 | 0.3585 | 91.39% | **2.33 ms** |
| **Spectral Clustering** | 12 | 0.6043 | 0.3242 | 88.66% | 45.2 ms |
| **Leiden** | 10 | 0.6290 | 0.2689 | 91.95% | 8.7 ms |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the CLI Benchmark Pipeline
```bash
python pipeline.py
```
Outputs benchmark tables, community map PNGs, and Gephi `.gexf` exports.

### 3. Run the Interactive Web App
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. Run in Jupyter Notebook
```bash
jupyter notebook Community_Detection_Project.ipynb
```

### 5. Run Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 🌐 Web Application Features

- **Dataset Selector**: Choose from 4 built-in benchmarks or upload custom CSV edge lists
- **Algorithm Picker**: Select from 5 algorithms with configurable parameters (resolution, target $k$)
- **Interactive Graph**: Force-directed Vis.js network with color-coded community nodes
- **Live Node Inspector**: Click any node to see degree, community assignment, confidence, neighbor distribution
- **EDA Statistics Panel**: Average degree, density, clustering coefficient, degree distribution histogram
- **Benchmark Modal**: Head-to-head comparison table + bar chart across all 5 algorithms
- **Export**: Screenshot graph as PNG, export to Gephi `.gexf`

---

## 📁 Repository Structure
```
├── app.py                            # Flask web application server
├── algorithms.py                     # 5 community detection algorithm wrappers
├── datasets.py                       # Real-world dataset loaders (SBM generators)
├── pipeline.py                       # CLI 5-step benchmark pipeline
├── templates/
│   └── index.html                    # Interactive web UI (glassmorphism + Vis.js)
├── static/
│   ├── css/style.css                 # Design system CSS
│   └── js/main.js                    # Frontend controller (legacy)
├── tests/
│   └── test_algorithms.py            # Pytest unit test suite
├── api/
│   └── index.py                      # Vercel serverless entry point
├── Community_Detection_Project.ipynb  # Jupyter Notebook for submission
├── sample_social_network.csv          # Sample 12-node CSV edge list
├── requirements.txt                   # Python dependencies
├── vercel.json                        # Vercel deployment config
├── render.yaml                        # Render deployment config
├── viva_presentation_guide.md         # Viva voce Q&A guide
├── PROJECT_PLAN.md                    # Full project development plan
└── README.md                          # This file
```

---

## 📐 Evaluation Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| **Modularity ($Q$)** | $[-0.5, 1.0]$ | Edge density inside vs. outside communities. $> 0.3$ = significant structure |
| **NMI** | $[0, 1]$ | Agreement with ground truth partition. 1.0 = perfect match |
| **ARI** | $[-1, 1]$ | Pair-wise clustering agreement, adjusted for chance |
| **Conductance ($\Phi$)** | $[0, 1]$ | Fraction of edges leaving community. Lower = better separation |

---

## 🔗 Key References

1. Blondel et al., *"Fast unfolding of communities in large networks"* (Louvain, 2008)
2. Girvan & Newman, *"Community structure in social and biological networks"* (PNAS 2002)
3. Traag et al., *"From Louvain to Leiden: guaranteeing well-connected communities"* (2019)
4. [NetworkX Community Detection Docs](https://networkx.org/documentation/stable/reference/algorithms/community.html)
5. [Stanford SNAP Datasets](https://snap.stanford.edu/data/)
