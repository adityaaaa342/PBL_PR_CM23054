"""
=============================================================================
Unit Tests for Community Detection in Social Networks
Tests: Algorithm wrappers, evaluation metrics, data loading, API endpoints
=============================================================================
"""

import os
import sys
import pytest
import networkx as nx
import numpy as np

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from algorithms import (
    detect_louvain,
    detect_girvan_newman,
    detect_label_propagation,
    detect_spectral,
    detect_leiden,
)
from datasets import (
    get_football_network,
    get_facebook_circles,
    get_coauthorship_network,
    get_student_network,
    load_custom_csv,
)
from pipeline import evaluate_communities


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def karate_graph():
    """Zachary's Karate Club — classic 34-node benchmark."""
    return nx.karate_club_graph()


@pytest.fixture
def small_sbm_graph():
    """Small stochastic block model with 3 known communities."""
    sizes = [10, 10, 10]
    probs = [[0.8, 0.05, 0.05], [0.05, 0.8, 0.05], [0.05, 0.05, 0.8]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    ground_truth = {}
    idx = 0
    for cid, size in enumerate(sizes):
        for _ in range(size):
            ground_truth[idx] = cid
            idx += 1
    return G, ground_truth


# ============================================================================
# Algorithm Output Schema Tests
# ============================================================================

EXPECTED_KEYS = {"name", "communities", "partition", "num_communities", "modularity", "time_ms"}


class TestAlgorithmOutputSchema:
    """All algorithm wrappers must return dicts with consistent keys."""

    def test_louvain_schema(self, karate_graph):
        result = detect_louvain(karate_graph)
        assert EXPECTED_KEYS.issubset(result.keys())

    def test_girvan_newman_schema(self, karate_graph):
        result = detect_girvan_newman(karate_graph, num_clusters=2)
        assert EXPECTED_KEYS.issubset(result.keys())

    def test_lpa_schema(self, karate_graph):
        result = detect_label_propagation(karate_graph)
        assert EXPECTED_KEYS.issubset(result.keys())

    def test_spectral_schema(self, karate_graph):
        result = detect_spectral(karate_graph, k=2)
        assert EXPECTED_KEYS.issubset(result.keys())

    def test_leiden_schema(self, karate_graph):
        result = detect_leiden(karate_graph)
        assert EXPECTED_KEYS.issubset(result.keys())


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================

class TestAlgorithmCorrectness:
    """Verify algorithms produce valid partitions and sensible metrics."""

    def _validate_partition(self, G, result):
        """Common partition validation checks."""
        partition = result["partition"]
        communities = result["communities"]

        # Every node must be assigned
        assert set(partition.keys()) == set(G.nodes())

        # Number of communities must match
        assert result["num_communities"] == len(communities)

        # All community members must cover all nodes
        all_members = set()
        for c in communities:
            all_members.update(c)
        assert all_members == set(G.nodes())

        # Modularity must be in valid range
        assert -0.5 <= result["modularity"] <= 1.0

        # Execution time must be non-negative
        assert result["time_ms"] >= 0

    def test_louvain_valid_partition(self, karate_graph):
        result = detect_louvain(karate_graph)
        self._validate_partition(karate_graph, result)
        # Karate club should yield modularity > 0.3
        assert result["modularity"] > 0.3

    def test_girvan_newman_valid_partition(self, karate_graph):
        result = detect_girvan_newman(karate_graph, num_clusters=2)
        self._validate_partition(karate_graph, result)

    def test_lpa_valid_partition(self, karate_graph):
        result = detect_label_propagation(karate_graph)
        self._validate_partition(karate_graph, result)

    def test_spectral_valid_partition(self, karate_graph):
        result = detect_spectral(karate_graph, k=2)
        self._validate_partition(karate_graph, result)
        assert result["num_communities"] == 2

    def test_leiden_valid_partition(self, karate_graph):
        result = detect_leiden(karate_graph)
        self._validate_partition(karate_graph, result)
        assert result["modularity"] > 0.3

    def test_sbm_high_quality(self, small_sbm_graph):
        """On a clear SBM, Louvain should find ~3 communities with high modularity."""
        G, _ = small_sbm_graph
        result = detect_louvain(G)
        assert result["num_communities"] >= 2
        assert result["modularity"] > 0.4


# ============================================================================
# Evaluation Metrics Tests
# ============================================================================

class TestEvaluationMetrics:
    """Test evaluation metric calculations."""

    def test_evaluate_with_ground_truth(self, small_sbm_graph):
        G, gt = small_sbm_graph
        result = {
            "algorithm": "Louvain",
            "communities": detect_louvain(G)["communities"],
            "partition": detect_louvain(G)["partition"],
            "num_communities": detect_louvain(G)["num_communities"],
            "time_ms": 1.0,
        }
        metrics = evaluate_communities(G, result, ground_truth=gt)
        assert "modularity" in metrics
        assert "conductance" in metrics
        assert metrics["nmi"] != "N/A"
        assert metrics["ari"] != "N/A"
        # NMI should be high for clear block structure
        assert float(metrics["nmi"]) > 0.5

    def test_evaluate_without_ground_truth(self, karate_graph):
        result = {
            "algorithm": "Louvain",
            "communities": detect_louvain(karate_graph)["communities"],
            "partition": detect_louvain(karate_graph)["partition"],
            "num_communities": detect_louvain(karate_graph)["num_communities"],
            "time_ms": 1.0,
        }
        metrics = evaluate_communities(karate_graph, result, ground_truth=None)
        assert metrics["nmi"] == "N/A"
        assert metrics["ari"] == "N/A"
        assert metrics["modularity"] > 0.0


# ============================================================================
# Dataset Loading Tests
# ============================================================================

class TestDatasetLoading:
    """Test that all dataset loaders return valid graphs and info dicts."""

    def _validate_dataset(self, G, info):
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
        assert "name" in info
        assert "ground_truth_k" in info

    def test_football_network(self):
        G, info = get_football_network()
        self._validate_dataset(G, info)
        assert G.number_of_nodes() == 115
        assert info["ground_truth_k"] == 12

    def test_facebook_circles(self):
        G, info = get_facebook_circles()
        self._validate_dataset(G, info)
        assert info["ground_truth_k"] == 4

    def test_coauthorship_network(self):
        G, info = get_coauthorship_network()
        self._validate_dataset(G, info)
        assert info["ground_truth_k"] == 3

    def test_student_network(self):
        G, info = get_student_network()
        self._validate_dataset(G, info)
        assert G.number_of_nodes() == 12
        assert info["ground_truth_k"] == 3

    def test_custom_csv_loader(self):
        csv_path = os.path.join(ROOT_DIR, "sample_social_network.csv")
        if os.path.exists(csv_path):
            G, info = load_custom_csv(csv_path)
            assert isinstance(G, nx.Graph)
            assert G.number_of_nodes() > 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test algorithms on degenerate or boundary graphs."""

    def test_single_edge_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        result = detect_louvain(G)
        assert result["num_communities"] >= 1

    def test_complete_graph(self):
        G = nx.complete_graph(10)
        result = detect_louvain(G)
        # Complete graph has no real community structure
        assert result["num_communities"] >= 1

    def test_path_graph(self):
        G = nx.path_graph(20)
        result = detect_label_propagation(G)
        assert result["num_communities"] >= 1

    def test_disconnected_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_edges_from([(10, 11), (11, 12), (12, 10)])
        result = detect_louvain(G)
        # Should find at least 2 communities for 2 disconnected triangles
        assert result["num_communities"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
