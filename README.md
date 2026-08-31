# Community Detection in Social Networks

**Course:** Pattern Recognition  
**Student:** Aditya Mahalle (USN: CM23054)  

---

## 📌 Project Overview
This project implements an automated, unsupervised **5-step pipeline** to discover natural, tightly-knit groups (communities) inside social networks using graph topology.

We evaluate and compare three fundamental community detection algorithms across real-world social networks with ground truth.

---

## 🔬 Algorithms Implemented
1. **Louvain Algorithm** — Modularity Maximization ($\mathcal{O}(N \log N)$)
2. **Girvan-Newman Algorithm** — Edge Betweenness Divisive ($\mathcal{O}(M^2 N)$)
3. **Label Propagation Algorithm (LPA)** — Near-linear Time Dynamic Diffusion ($\mathcal{O}(M + N)$)

---

## 📂 Real-World Datasets
- **American College Football Network** (115 teams, 613 games, 12 Conferences) — *Girvan & Newman (2002)*
- **Facebook Social Circles** (78 users, 413 friendships, 4 Social Circles) — *SNAP*
- **Academic Co-Authorship Network** (120 authors, 995 papers, 3 Sub-Disciplines) — *DBLP*

---

## 📊 Benchmark Results (American College Football Network)

| Algorithm | Method Category | Communities Found | Modularity ($Q$) | Conductance | NMI Accuracy | Runtime (ms) |
|---|---|---|---|---|---|---|
| **Louvain** | Modularity Maximization | 10 | **0.6290** | 0.2689 | **0.9195** | 12.11 ms |
| **Girvan-Newman** | Edge Betweenness Divisive | 12 | 0.6043 | 0.3242 | 0.8866 | 3090.67 ms |
| **Label Propagation** | Dynamic Consensus | 14 | 0.5972 | 0.3585 | 0.9139 | **4.23 ms** |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the 5-Step Pipeline
```bash
python pipeline.py
```

### 3. Run in Jupyter Notebook
```bash
jupyter notebook Community_Detection_Project.ipynb
```

---

## 📁 Repository Structure
```
├── pipeline.py                       # Main 5-step pipeline script
├── algorithms.py                     # Standalone algorithm implementations
├── datasets.py                       # Real-world dataset loaders
├── Community_Detection_Project.ipynb # Complete Jupyter Notebook for submission
├── requirements.txt                  # Python dependencies
├── viva_presentation_guide.md        # Presentation script & Viva Voce Q&A
├── football_louvain.png              # Output community map (Louvain)
├── football_gn.png                   # Output community map (Girvan-Newman)
├── football_lpa.png                  # Output community map (Label Propagation)
└── README.md                         # Project documentation
```
