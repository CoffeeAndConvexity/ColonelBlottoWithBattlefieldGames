from typing import List


class DagStructure:
    """
    Note:   Vertex === Infoset
            Edges === Sequences
            Parents === Predecesor Edge/Sequence
        
        Crucial to understand that each edge can have multiple children vertices.
    """

    def __init__(self):
        """
        Initializes an empty directed acyclic graph (DAG).
        """

        self.num_sequences = 1  # Number of sequences in the DAG, begin with empty sequence.
        self.num_infosets = 0  # Number of infosets in the DAG
        
        # Contains one number for each infoset, which is the number of actions available in that infoset.
        self.infoset_num_actions = []

        # Contains one number for each infoset, which is the starting sequence ID for that infoset.
        self.infoset_start_seq_id = []

        # Contains one *list* for each infoset, which contains all the parents' sequence ID for that infoset.
        self.infoset_parent_seq_id = []

        # Contains one *list* for each sequence, which contains the child infoset id for that sequence.
        self.seq_id_child_infoset_id = [[]] # Starts with empty list for empty sequence.

        # Mappings for infoset names to IDs and vice versa.
        self.infoset_name_to_id = dict()
        self.infoset_id_to_name = []


    def add_infoset(self, 
                    parent_seq_ids: List[int], 
                    num_actions: int, 
                    infoset_name: str):
        """
        Adds a new vertex to the DAG.

        Args:
            parent_seq_ids: List[int]: A list of sequence IDs representing the parent sequences of the new infoset.
            num_actions: int: The number of actions available in the new infoset.
        """

        assert num_actions > 0, "Number of actions must be greater than zero."
        if infoset_name in self.infoset_name_to_id:
            raise ValueError(f"Infoset {infoset_name} already exists in the graph.")
        infoset_id = self.num_infosets
        self.infoset_id_to_name.append(infoset_name)
        self.infoset_name_to_id[infoset_name] = infoset_id
        self.num_infosets += 1

        assert len(parent_seq_ids) > 0, "Parent sequence IDs cannot be empty."


        self.infoset_num_actions.append(num_actions)
        self.infoset_parent_seq_id.append(parent_seq_ids.copy())
        self.infoset_start_seq_id.append(self.num_sequences)

        # Add in sequence ids for actions belonging to new infoset. Should be empty list
        # since this has a dag structure.       
        for seq_id in range(self.num_sequences, self.num_sequences + num_actions):
            self.seq_id_child_infoset_id.append([])
        self.num_sequences += num_actions
        
        for parent_seq_id in parent_seq_ids:
            self.seq_id_child_infoset_id[parent_seq_id].append(infoset_id)

    def get_infoset_infoset_children(self):
        for infoset_id in range(self.num_infosets):
            infoset_children = []
            for seq_id in self.infoset_parent_seq_id[infoset_id]:
                infoset_children.extend(self.seq_id_child_infoset_id[seq_id])
            yield infoset_children
    
def unit_test():
    dag = DagStructure()
    dag.add_infoset([0], 3, "Infoset1")
    assert dag.num_infosets == 1
    assert dag.num_sequences == 4
    assert dag.infoset_num_actions == [3] # probably wrong.
    assert dag.infoset_parent_seq_id == [[0]]
    assert dag.infoset_start_seq_id == [1]
    assert dag.seq_id_child_infoset_id == [[0], [], [], []]
    assert dag.infoset_id_to_name == ["Infoset1"]

    dag.add_infoset([1, 3], 2, "Infoset2")
    assert dag.num_infosets == 2
    assert dag.num_sequences == 6
    assert dag.infoset_num_actions == [3, 2]
    assert dag.infoset_parent_seq_id == [[0], [1, 3]]
    assert dag.infoset_start_seq_id == [1, 4]
    assert dag.seq_id_child_infoset_id == [[0], [1], [], [1], [], []]
    
    dag.add_infoset([1, 3], 2, "Infoset3")
    assert dag.num_infosets == 3
    assert dag.num_sequences == 8
    assert dag.infoset_num_actions == [3, 2, 2]
    assert dag.infoset_parent_seq_id == [[0], [1, 3], [1, 3]]
    assert dag.infoset_start_seq_id == [1, 4, 6]
    assert dag.seq_id_child_infoset_id == [[0], [1, 2], [], [1, 2], [], [], [], []]

    dag.add_infoset([2, 7], 2, "Infoset4")
    assert dag.num_infosets == 4
    assert dag.num_sequences == 10
    assert dag.infoset_num_actions == [3, 2, 2, 2]
    assert dag.infoset_parent_seq_id == [[0], [1, 3], [1, 3], [2, 7]]
    assert dag.infoset_start_seq_id == [1, 4, 6, 8]
    assert dag.seq_id_child_infoset_id == [[0], [1, 2], [3], [1, 2], [], [], [], [3], [], []]

    print("Number of sequences:", dag.num_sequences)
    print("Number of infosets:", dag.num_infosets)
    print("Infoset names to IDs:", dag.infoset_name_to_id)
    print("Infoset IDs to names:", dag.infoset_id_to_name)

if __name__ == '__main__':
    unit_test()