"""
Same as basic_blotto.py but with a more general structure of battlefield games.

TODO: refactor the code to have basic blotto as a subclass of this class.
"""


from typing import Dict, List, Tuple
from online_learning.dag_structure import DagStructure
from online_learning.regret_matching import RegretMatching
from online_learning.dag_treeplex import DagTreeplex
from online_learning.dag_regret_minimizer import DagGame
import numpy as np 
import copy

class BayesianBattlefieldGame(object):
    '''
    This is for a single battlefield!
    '''
    def __init__(self, 
                 max_soldiers_p1: int,
                 max_soldiers_p2: int,
                 num_actions_p1: List[int], 
                 num_actions_p2: List[int],
                 payoff_matrices: List[List[np.ndarray]],
                 ):
        """
        Initializes a Bayesian battlefield game.

        Args:
            max_soldiers_p1 (int): Maximum number of soldiers player 1 can allocate.
            max_soldiers_p2 (int): Maximum number of soldiers player 2 can allocate.
            num_actions_p1 (List[int]): List of possible actions for player 1 in each type.
            num_actions_p2 (List[int]): List of possible actions for player 2 in each type.
            payoff_matrices (List[List[np.ndarray]]): Payoff matrices for each pairs of types.
        """
        self.max_soldiers_p1 = max_soldiers_p1
        self.max_soldiers_p2 = max_soldiers_p2
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.payoff_matrices = payoff_matrices

        # Check that the payoff matrices are of the correct size.
        print(len(payoff_matrices), max_soldiers_p1 + 1)
        assert len(payoff_matrices) == max_soldiers_p1 + 1
        assert all(len(payoff_matrices[i]) == max_soldiers_p2 + 1 for i in range(max_soldiers_p1 + 1))
        for i in range(max_soldiers_p1 + 1):
            for j in range(max_soldiers_p2 + 1):
                assert payoff_matrices[i][j].shape == (num_actions_p1[i], num_actions_p2[j]), \
                    f"Payoff matrix for ({i}, {j}) does not match the number of actions."
                
                

class GeneralizedBBBlottoGame(DagGame):
    def __init__(self, num_battles: int, 
                 num_soldiers : Tuple[int, int],
                 battlefield_bayesian_games: List[BayesianBattlefieldGame],
                 ):
        """
        Initializes the Battlefield-Bayesian Blotto game with a specified number of players and battles.

        Note we allow for players to not use all soldiers.

        Args:
            num_battles (int): Number of battles in the game.
            battlefield_bayesian_payoffs (List): List of bayesian noralized payoffs for each battlefield.
        """
        self.num_battles = num_battles
        self.battlefield_bayesian_games = battlefield_bayesian_games
        self.num_soldiers_p1, self.num_soldiers_p2 = num_soldiers
        num_soldiers_p1, num_soldiers_p2 = num_soldiers

        assert len(battlefield_bayesian_games) == num_battles
        
        # Get action sizes for each player, for each about of soliders played.
        action_sizes_p1 = []
        action_sizes_p2 = []
        for battle_id in range(num_battles):
            action_sizes_p1.append([])
            action_sizes_p2.append([])
            for num_soldiers_used_p1 in range(num_soldiers_p1+1):
                num_actions = battlefield_bayesian_games[battle_id].num_actions_p1[num_soldiers_used_p1]
                action_sizes_p1[-1].append(num_actions)
            for num_soldiers_used_p2 in range(num_soldiers_p2+1):
                num_actions = battlefield_bayesian_games[battle_id].num_actions_p2[num_soldiers_used_p2]
                action_sizes_p2[-1].append(num_actions)


        # Generate DAGs for each player.
        dag_p1 = GeneralizedBBBlottoGame.generate_dag(num_battles, num_soldiers_p1, action_sizes_p1)
        dag_p2 = GeneralizedBBBlottoGame.generate_dag(num_battles, num_soldiers_p2, action_sizes_p2)

        # Generate payoffs for each player. 
        leaves = GeneralizedBBBlottoGame.generate_sparse_payoffs(dag_p1, 
                                                    dag_p2, 
                                                    num_battles, 
                                                    num_soldiers_p1, 
                                                    num_soldiers_p2,
                                                    battlefield_bayesian_games)

        super().__init__(dag_p1, dag_p2, leaves)       


    def generate_dag(num_battles: int, 
                     num_soldiers: int,
                     num_actions_per_bf_per_soldiers : List[List[int]]):
        """

        Additional speedup:
        --------
        We add in one dummy infoset for each battle for each number of soldiers used. 
        This is used to "fake" payoffs and make the payoff matrix a lot sparser.
        """

        dag = DagStructure()

        # First battlefield is special.
        dag.add_infoset([0], num_soldiers+1, (0, num_soldiers))

        # Add in main infosets
        for battle_id in range(1, num_battles):
            for num_soldiers_left in range(num_soldiers+1):
                # Get parent sequences.
                par_seq_ids = []
                for prev_num_soldiers in range(num_soldiers_left, num_soldiers+1):
                    num_soldiers_used = prev_num_soldiers - num_soldiers_left
                    assert num_soldiers_used >= 0

                    par_infoset_name = (battle_id-1, prev_num_soldiers)
                    if par_infoset_name not in dag.infoset_name_to_id:
                        assert battle_id == 1
                    else:
                        par_infoset_id = dag.infoset_name_to_id[par_infoset_name]
                        par_seq_id = dag.infoset_start_seq_id[par_infoset_id] + num_soldiers_used
                        par_seq_ids.append(par_seq_id)
                        

                dag.add_infoset(par_seq_ids, num_soldiers_left+1, (battle_id, num_soldiers_left))

        # Add in dummy infosets
        for battle_id in range(num_battles):
            for num_soldiers_used in range(num_soldiers+1):
                par_seq_ids = []
                
                # Infoset name is ('d', battle_id, num_soldiers_used)
                # Special 'd' marker to distinguish this name from the regular infoset.
                new_infoset_name = ('d', battle_id, num_soldiers_used)

                # Iterate over all possible parent sequences
                for initial_infoset in range(num_soldiers_used, num_soldiers+1):
                    par_infoset_name = (battle_id, initial_infoset)
                    if par_infoset_name not in dag.infoset_name_to_id:
                        assert battle_id == 0
                    else:
                        infoset_id = dag.infoset_name_to_id[par_infoset_name]
                        seq_id = dag.infoset_start_seq_id[infoset_id] + num_soldiers_used
                        par_seq_ids.append(seq_id)
                
                # Create new dummy infoset
                dag.add_infoset(par_seq_ids, 
                                num_actions_per_bf_per_soldiers[battle_id][num_soldiers_used], 
                                new_infoset_name)
                
        # print('----------------------------------------------')
        return dag

    def generate_sparse_payoffs(dag_p1: DagStructure, 
                                dag_p2: DagStructure,
                                num_battles: int, 
                                num_soldiers_p1: int,
                                num_soldiers_p2: int,
                                battlefield_bayesian_games: List[BayesianBattlefieldGame]):
        """
        Generates sparse payoffs for the Blotto game.

        Args:
            dag_p1 (DagStructure): DAG structure for player 1.
            dag_p2 (DagStructure): DAG structure for player 2.
            num_battles (int): Number of battles in the game.
            num_soldiers_p1 (int): Number of soldiers for player 1.
            num_soldiers_p2 (int): Number of soldiers for player 2.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]: Payoffs for both players.
        """

        leaves = dict()

         # Iterate through all dummy infosets in the DAG
        for battle_id in range(num_battles):
            bbg = battlefield_bayesian_games[battle_id]
            for num_soldiers_used_p1 in range(num_soldiers_p1+1):
                for num_soldiers_used_p2 in range(num_soldiers_p2+1):
                    p1_dummy_infoset = dag_p1.infoset_name_to_id[('d', battle_id, num_soldiers_used_p1)]
                    p2_dummy_infoset = dag_p2.infoset_name_to_id[('d', battle_id, num_soldiers_used_p2)]
                    
                    submatrix_game = bbg.payoff_matrices[num_soldiers_used_p1][num_soldiers_used_p2]
                    assert submatrix_game.shape == (bbg.num_actions_p1[num_soldiers_used_p1], bbg.num_actions_p2[num_soldiers_used_p2])

                    p1_start_seq_id = dag_p1.infoset_start_seq_id[p1_dummy_infoset]
                    p2_start_seq_id = dag_p2.infoset_start_seq_id[p2_dummy_infoset]

                    for action_id_p1 in range(bbg.num_actions_p1[num_soldiers_p1]):
                        for action_id_p2 in range(bbg.num_actions_p2[num_soldiers_p2]):
                            payoff = submatrix_game[action_id_p1, action_id_p2]
                            leaves[(p1_start_seq_id + action_id_p1, p2_start_seq_id + action_id_p2)] = payoff
        
        return leaves

def unit_test():
    BlottoGame(3, (5, 3), [1.0, 1.0, 1.0])

if __name__ == "__main__":
    unit_test()
