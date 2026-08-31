# Community Detection in Social Networks
## Presentation & Viva Voce Preparation Guide

> **Course**: Pattern Recognition  
> **Topic**: Community Detection in Social Networks  
> **Matching Slides**: Slides 1 to 10 from your presentation  

---

## 📌 Slide-by-Slide Explanation & Speech Script

### Slide 1: Title Slide
- **What to say**: *"Good morning/afternoon professors. Today I am presenting my project on **Community Detection in Social Networks**, which focuses on identifying natural, tightly-knit clusters of users inside complex graph networks."*

---

### Slide 2: Problem Statement
- **Key Concepts**:
  - Social graphs have thousands or millions of interactions (friendships, follows, collaborations).
  - Communities are not labeled in advance (unsupervised pattern recognition).
  - Manual inspection is impossible at scale.
- **What to say**: *"In real-world social networks, community boundaries are completely unlabeled. Our goal is to develop an automated pipeline that can uncover these hidden groups based solely on the topology of who interacts with whom."*

---

### Slide 3 & 4: Project Pipeline & Data Collection
- **The 5-Step Pipeline**:
  1. **Collect Data**: Load datasets (Zachary's Karate Club, Facebook Ego Networks, Bottlenose Dolphins).
  2. **Build & Clean Graph**: Construct graphs in `NetworkX`, remove self-loops and isolated nodes.
  3. **Detect Communities**: Apply Louvain, Girvan-Newman, and Label Propagation.
  4. **Evaluate**: Score partitions using Modularity ($Q$), Conductance, NMI, and ARI.
  5. **Visualize & Interpret**: Generate colored community maps and export to Gephi.

---

### Slide 5 & 6: The 3 Core Community Detection Algorithms

| Algorithm | Method / Core Idea | Time Complexity | Strength / Trade-off |
|---|---|---|---|
| **Louvain** | **Modularity Optimization**: Greedily moves nodes between communities to maximize Newman's modularity score $Q$. | $\mathcal{O}(N \log N)$ | Very fast, scalable, and the standard baseline for large networks. |
| **Girvan–Newman** | **Edge Betweenness Centrality**: Iteratively identifies and cuts the "bridge" edges connecting different communities. | $\mathcal{O}(M^2 N)$ | Highly intuitive hierarchical division, but computationally slower. |
| **Label Propagation (LPA)** | **Dynamic Consensus**: Each node iteratively adopts the majority label of its direct neighbors. | $\mathcal{O}(M + N)$ | Near-linear time, runs almost instantaneously on large graphs. |

---

### Slide 7 & 8: Tech Stack & Evaluation Metrics

#### 1. Evaluation Metrics:
- **Modularity ($Q$)**: Measures whether connections inside communities are denser than expected in a random graph. (Ranges from $-0.5$ to $1.0$; scores $>0.3$ represent strong community structure).
- **Conductance ($\Phi$)**: Ratio of edges cutting outside the community vs total internal edges (lower conductance = cleaner community isolation).
- **Normalized Mutual Information (NMI)**: Compares detected partitions against known ground-truth factions (Score $0.0$ to $1.0$, $1.0$ is perfect match).
- **Adjusted Rand Index (ARI)**: Pairwise clustering similarity score against ground truth.

#### 2. Tech Stack:
- **Language**: Python
- **Graph Processing**: `NetworkX`, `NumPy`, `Pandas`
- **Evaluation**: `scikit-learn` (NMI, ARI)
- **Visualization**: `Matplotlib`, `Gephi` (GEXF export)

---

### Slide 9 & 10: Expected Outcomes & Results

#### Actual Benchmark Results on Zachary's Karate Club:
```
===========================================================================
  ALGORITHM BENCHMARK RESULTS (Zachary's Karate Club: 34 Nodes, 78 Edges)
===========================================================================
        Algorithm  Communities  Modularity Q  Conductance     NMI   Time (ms)
          Louvain            4        0.4266       0.3528  0.5942     2.69 ms
    Girvan-Newman            5        0.3850       0.4535  0.4851    60.07 ms
Label Propagation            4        0.4366       0.2970  0.5056     0.97 ms
===========================================================================
```

---

## 🎯 Top Professor Viva Questions & Exact Answers

### Q1: What is Newman's Modularity ($Q$)?
> **Answer**: *"Modularity is a fitness metric that quantifies the quality of a network partition. It measures the fraction of edges falling inside communities minus the expected fraction of edges if edges were distributed randomly. A higher score (typically $>0.3$) indicates dense internal clusters with sparse connections between them."*

### Q2: Why is Girvan-Newman slower than Louvain?
> **Answer**: *"Girvan-Newman calculates edge betweenness centrality for all edges in every iteration using shortest-path traversals ($\mathcal{O}(V \cdot E)$), repeating this each time an edge is removed. In contrast, Louvain uses local heuristic greedy moves that only inspect immediate neighbor nodes, running in near $\mathcal{O}(V \log V)$ time."*

### Q3: What happens in Label Propagation if there is a tie?
> **Answer**: *"If a node's neighbors have an equal count of two different labels, the algorithm breaks ties uniformly at random or retains the node's current label."*

### Q4: How do you evaluate a network that has no ground truth?
> **Answer**: *"For unlabeled networks, we evaluate structural quality using Modularity ($Q$), Conductance ($\Phi$), and coverage. When ground truth is available (such as the 2 factions in the Karate Club), we use Normalized Mutual Information (NMI) and ARI."*

---

## 📁 File Reference
- **5-Step Python Pipeline**: [`pipeline.py`](file:///C:/Users/umesh/.gemini/antigravity-ide/scratch/social_network_community_detection/pipeline.py)
- **Jupyter Notebook**: [`Community_Detection_Project.ipynb`](file:///C:/Users/umesh/.gemini/antigravity-ide/scratch/social_network_community_detection/Community_Detection_Project.ipynb)
- **Visual Community Images**: `karate_louvain.png`, `karate_girvan_newman.png`, `karate_lpa.png`
- **Gephi Export**: `network_communities.gexf`
