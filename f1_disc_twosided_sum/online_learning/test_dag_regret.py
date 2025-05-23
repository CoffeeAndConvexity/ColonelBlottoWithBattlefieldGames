import unittest
import numpy as np
from online_learning.regret_matching import RegretMatching
from online_learning.dag_structure import DagStructure
from online_learning.dag_regret_minimizer import DagRegretMinimizer

class TestDagRegretMinimizer(unittest.TestCase):
    def test_initialization(self):
        """Test if DagRegretMinimizer initializes correctly."""
        dag = DagStructure()
        dag.add_vertex(0, [])
        dag.add_vertex(1, [0])
        dag.add_vertex(2, [0, 1])
        drm = DagRegretMinimizer(dag)

        self.assertEqual(drm.dag_structure.get_num_vertices(), 3)
        self.assertEqual(drm.dag_structure.get_num_edges(), 3)

    def test_update_and_recommend(self):
        """Test if update and recommend work correctly in DagRegretMinimizer."""
        dag = DagStructure()
        dag.add_vertex(0, [])
        dag.add_vertex(1, [0])
        dag.add_vertex(2, [0, 1])
        drm = DagRegretMinimizer(dag, num_actions=3)

        rewards = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]), 2: np.array([0, 0, 1])}
        last_strategies = {0: np.array([0.5, 0.25, 0.25]), 1: np.array([0.33, 0.33, 0.34]), 2: np.array([0.25, 0.25, 0.5])}

        drm.update_regrets(rewards, last_strategies)
        for vertex in range(3):
            self.assertTrue(np.all(drm.regrets[vertex] >= 0))

if __name__ == "__main__":
    unittest.main()