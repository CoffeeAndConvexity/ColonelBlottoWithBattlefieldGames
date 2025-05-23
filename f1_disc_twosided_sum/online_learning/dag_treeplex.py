from online_learning.dag_structure import DagStructure
from online_learning.regret_matching import RegretMatching
import numpy as np

class DagTreeplex(object):
    def __init__(self, dag_structure: DagStructure, init = None):
        """
        Initializes the DagTreeplex with a given DagStructure.

        Args:
            dag_structure: An instance of DagStructure representing the directed acyclic graph.
        """
        self.dag_structure = dag_structure

        if init is None:
            self.treeplex_data = np.zeros(dag_structure.num_sequences, dtype=np.float64)
        else:
            assert len(init) == dag_structure.num_sequences, "Initialization array must match the number of sequences."
            self.treeplex_data = init

    def best_response_to_reward_vector(self, reward_vector: np.ndarray, inplace = False):
        assert reward_vector.size == self.dag_structure.num_sequences, "Reward vector size must match the number of sequences."
        
        # If inplace is False, create a copy of the reward vector
        if inplace == False:
            rewards = reward_vector.copy()
        else:
            rewards = reward_vector
        
        ret = np.zeros(self.dag_structure.num_sequences, dtype=np.float64)

        for infoset_id in reversed(range(self.dag_structure.num_infosets)):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]

            # Get the best action for the current infoset
            best_action = np.argmax(rewards[start_seq_id: start_seq_id + num_actions])
            best_action_seq_id = start_seq_id + best_action
            ret[best_action_seq_id] = 1.0

            # Update the rewards for the parent sequences
            for parent_seq_id in self.dag_structure.infoset_parent_seq_id[infoset_id]:
                rewards[parent_seq_id] += rewards[best_action_seq_id]

        ret[0] = 1.0 # Empty sequence defaults to 1
        beh = DagTreeplex(self.dag_structure, ret)
        beh.convert_beh_to_seq() # Usually not needed, but in some special cases, sure.
        return beh

    def fill_with_unif_seq_form(self):
        """
        Fills the treeplex data structure with a uniform distribution in sequence form.
        """
        
        self.treeplex_data = np.zeros(self.dag_structure.num_sequences, dtype=np.float64)
        self.treeplex_data[0] = 1.0

        for infoset_id in range(self.dag_structure.num_infosets):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]
            parent_seq_ids = self.dag_structure.infoset_parent_seq_id[infoset_id]

            parent_mass = sum(self.treeplex_data[seq_id] for seq_id in parent_seq_ids)

            for seq_id in range(start_seq_id, start_seq_id + num_actions):
                self.treeplex_data[seq_id] = parent_mass / num_actions

    def fill_with_unif_beh_form(self):
        """
        Fills the treeplex data structure with a uniform distribution in behavior form.
        """
        self.treeplex_data[0] = 1.0
        for infoset_id in range(self.dag_structure.num_infosets):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]

            for seq_id in range(start_seq_id, start_seq_id + num_actions):
                self.treeplex_data[seq_id] = 1.0 / num_actions

    def convert_seq_to_beh(self):
        """
        Converts the treeplex data from sequence form to behavior form.
        """
        assert self.treeplex_data[0] == 1.0
        self.treeplex_data[0] = 1.0
        for infoset_id in range(self.dag_structure.num_infosets):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]

            total_child_mass = np.sum(self.treeplex_data[start_seq_id: (start_seq_id + num_actions)]) # sum(self.treeplex_data[seq_id] for seq_id in range(start_seq_id, start_seq_id + num_actions))

            if total_child_mass == 0:
                # If sequence form is zero, set behavior form to uniform distribution
                for seq_id in range(start_seq_id, start_seq_id + num_actions):
                    self.treeplex_data[seq_id] = 1./num_actions
            else:
                for seq_id in range(start_seq_id, start_seq_id + num_actions):
                    self.treeplex_data[seq_id] = self.treeplex_data[seq_id] / total_child_mass

    def convert_beh_to_seq(self):
        """
        Converts the treeplex data from behavior form to sequence form by top down traversal.
        """
        self.treeplex_data[0] = 1.0
        for infoset_id in range(self.dag_structure.num_infosets):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]
            parent_seq_ids = self.dag_structure.infoset_parent_seq_id[infoset_id]

            parent_mass = sum(self.treeplex_data[seq_id] for seq_id in parent_seq_ids)

            for seq_id in range(start_seq_id, start_seq_id + num_actions):
                self.treeplex_data[seq_id] = parent_mass * self.treeplex_data[seq_id]

    def __str__(self):
        """
        Returns a string representation of the treeplex data.
        """
        s = []
        for infoset_id in range(self.dag_structure.num_infosets):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]
            s.append(f"{self.dag_structure.infoset_id_to_name[infoset_id]}: {self.treeplex_data[start_seq_id: start_seq_id + num_actions]}")
        return "\n".join(s)

    def __repr__(self):
        """
        Returns a string representation of the treeplex data.
        """
        return str(self.treeplex_data)

def unit_test():
    import copy

    dag = DagStructure()
    dag.add_infoset([0], 3, "Infoset1")
    dag.add_infoset([1, 3], 2, "Infoset2")
    dag.add_infoset([1, 3], 2, "Infoset3")
    dag.add_infoset([2, 7], 2, "Infoset4")

    unif_seq_form = DagTreeplex(dag)
    unif_seq_form.fill_with_unif_seq_form()

    unif_beh_form = DagTreeplex(dag)
    unif_beh_form.fill_with_unif_beh_form()

    print(unif_seq_form.treeplex_data)
    print(unif_beh_form.treeplex_data)

    unif_beh_form_cp = copy.deepcopy(unif_beh_form)
    unif_beh_form_cp.convert_beh_to_seq()
    np.testing.assert_almost_equal(unif_beh_form_cp.treeplex_data, unif_seq_form.treeplex_data)

    unif_seq_form_cp = copy.deepcopy(unif_seq_form)
    unif_seq_form_cp.convert_seq_to_beh()
    np.testing.assert_almost_equal(unif_seq_form_cp.treeplex_data, unif_beh_form.treeplex_data)

    print("Uniform Sequence Form:")
    print(unif_seq_form)
    print("Uniform Behavior Form:")
    print(unif_beh_form)

if __name__ == "__main__":
    unit_test()