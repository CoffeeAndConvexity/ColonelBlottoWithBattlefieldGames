from typing import Dict, Tuple
from online_learning.dag_structure import DagStructure
from online_learning.regret_matching import RegretMatching
from online_learning.dag_treeplex import DagTreeplex
import numpy as np 

class DagGame(object):
    def __init__(self, dag_structure_pl1: DagTreeplex, 
                 dag_structure_pl2: DagTreeplex,
                 leaves: Dict[Tuple[int, int], float]):
        
        self.dag_structure_pl1 = dag_structure_pl1
        self.dag_structure_pl2 = dag_structure_pl2

        self.leaves = leaves

    def saddle_point_gap(self, strategy_p1: DagTreeplex, strategy_p2: DagTreeplex):
        """
        Compute the saddle point gap for the given strategies.

        Args:
            strategy_p1 (DagTreeplex): The strategy for player 1.
            strategy_p2 (DagTreeplex): The strategy for player 2.

        Returns:
            float: The saddle point gap.
        """
        reward_for_p1, reward_for_p2 = self.compute_reward_vectors(strategy_p1, strategy_p2)
        br_p1 = strategy_p1.best_response_to_reward_vector(reward_for_p1)
        br_p2 = strategy_p2.best_response_to_reward_vector(reward_for_p2)

        val_p1_deviate = self.evaluate(br_p1, strategy_p2)
        val_p2_deviate = self.evaluate(strategy_p1, br_p2)
        
        assert val_p1_deviate - val_p2_deviate >= 1e-6, "Saddle point gap should not be negative"
        return val_p1_deviate - val_p2_deviate

    def evaluate(self, strategy_p1: DagTreeplex, strategy_p2: DagTreeplex):
        """
        Evaluate the game using the given strategies for both players.

        Args:
            strategy_p1 (DagTreeplex): The strategy for player 1.
            strategy_p2 (DagTreeplex): The strategy for player 2.

        Returns:
            float: The payoff for player 1.
        """
        payoff = 0.0
        for (seq_id1, seq_id2), reward in self.leaves.items():
            payoff += reward * strategy_p1.treeplex_data[seq_id1] * strategy_p2.treeplex_data[seq_id2]
        return payoff

    def compute_reward_vectors(self, strategy_p1: DagTreeplex, strategy_p2: DagTreeplex):
        """
        Computes the reward vectors for two players based on their strategies in a DAG game.

        Args:
            strategy_p1 (DagTreeplex): The strategy of player 1, represented as a DagTreeplex object.
            strategy_p2 (DagTreeplex): The strategy of player 2, represented as a DagTreeplex object.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the reward vector for player 1 and the reward vector for player 2.

        Notes:
            - The reward vector for player 1 is computed by summing the rewards at the leaves of the DAG,
              weighted by the strategy of player 2 for the corresponding sequences.
            - The reward vector for player 2 is computed similarly, but the rewards are subtracted and
              weighted by the strategy of player 1 for the corresponding sequences.
            - This method assumes that the `leaves` attribute of the `DagGame` object contains the reward
              information as a dictionary mapping sequence pairs to rewards.
        """
        
        reward_vector_p1 = np.zeros(self.dag_structure_pl1.num_sequences)
        reward_vector_p2 = np.zeros(self.dag_structure_pl2.num_sequences)
        for (seq_id1, seq_id2), reward in self.leaves.items():
            reward_vector_p1[seq_id1] += reward * strategy_p2.treeplex_data[seq_id2]
            reward_vector_p2[seq_id2] -= reward * strategy_p1.treeplex_data[seq_id1]

        return reward_vector_p1, reward_vector_p2
        
class DagRegretMinimizer:
    def __init__(self, dag_structure: DagStructure):
        """
        Initializes the DagRegretMinimizer with a given DagStructure.

        Args:
            dag_structure: An instance of DagStructure representing the directed acyclic graph.
        """
        self.dag_structure = dag_structure  # Store the immutable DagStructure
        self.regret_minimizers = []

        self.last_strategy = None

        # Create a list of RegretMatching instances or None for each vertex
        for infoset_id in range(dag_structure.num_infosets):
            self.regret_minimizers.append(RegretMatching(dag_structure.infoset_num_actions[infoset_id]))
        
    def observe_rewards(self, orig_rewards, 
                        last_strategy: DagTreeplex = None, 
                        inplace_rewards = False):
        """
        Observes the rewards for the current strategy and updates the regrets.

        Args:
            orig_rewards (DagTreeplex): The observed rewards for the current strategy.
            last_strategy (DagTreeplex): The last strategy played.
        """

        if inplace_rewards:
            rewards = orig_rewards
        else:
            rewards = orig_rewards.copy()

        if last_strategy is None:
            last_strategy = self.last_strategy

        for infoset_id in reversed(range(self.dag_structure.num_infosets)):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]
            parent_seq_ids = self.dag_structure.infoset_parent_seq_id[infoset_id]

            # Extract the rewards for the current infoset
            observed_rewards = rewards[start_seq_id: start_seq_id + num_actions]

            # Update the regrets for that particular simplex
            self.regret_minimizers[infoset_id].update_regrets(observed_rewards)
            
            # Reward that we would have gotten using the behavioral strategy.
            # We compute this using the normalized last strategy.
            """
            total_child_mass = np.sum(last_strategy.treeplex_data[start_seq_id : start_seq_id + num_actions])

            if total_child_mass == 0.0:
                # Assume uniform strategy was played.
                # normalized_reward = sum(rewards[seq_id] for seq_id in range(start_seq_id, start_seq_id + num_actions)) / num_actions
                normalized_reward = np.inner(rewards[start_seq_id : start_seq_id + num_actions], 
                                            self.regret_minimizers[infoset_id].last_strategy)
            else:
                normalized_reward = np.inner(rewards[start_seq_id : start_seq_id + num_actions], 
                                            last_strategy.treeplex_data[start_seq_id : start_seq_id + num_actions]) / total_child_mass
                # normalized_reward = sum(rewards.treeplex_data[seq_id] * last_strategy.treeplex_data[seq_id] for seq_id in range(start_seq_id, start_seq_id + num_actions)) / total_child_mass
            """
            normalized_reward = np.inner(rewards[start_seq_id : start_seq_id + num_actions], 
                                        self.regret_minimizers[infoset_id].last_strategy)
                
            # Push rewards upwards to parent sequences.
            for parent_seq_id in parent_seq_ids:
                rewards[parent_seq_id] += normalized_reward

    def recommend(self):
        """
        Generate a recommendation in *sequence* form.

        Returns:
            DagTreeplex: A sequence form treeplex strategy.
        """
        recommendations = DagTreeplex(self.dag_structure)
        recommendations.treeplex_data[0] = 1.0

        for infoset_id in range(self.dag_structure.num_infosets):
            num_actions = self.dag_structure.infoset_num_actions[infoset_id]
            start_seq_id = self.dag_structure.infoset_start_seq_id[infoset_id]
            
            # Get the current strategy for the infoset
            strategy = self.regret_minimizers[infoset_id].recommend()

            # Fill the recommendations for this infoset
            for seq_id in range(start_seq_id, start_seq_id + num_actions):
                recommendations.treeplex_data[seq_id] = strategy[seq_id - start_seq_id]

        recommendations.convert_beh_to_seq()

        self.last_strategy = recommendations

        return recommendations
    
    def solve_dag_game(dag_game: DagGame, iterations=10000):
        """
        Solve the DAG game using regret minimization.

        Args:
            reward_fn (callable): A function that takes a pair of sequences and outputs the reward.
            iterations (int): Number of iterations for regret minimization.
        """

        player1 = DagRegretMinimizer(dag_game.dag_structure_pl1)
        player2 = DagRegretMinimizer(dag_game.dag_structure_pl2)

        cumulative_strategy_p1 = np.zeros(dag_game.dag_structure_pl1.num_sequences)
        cumulative_strategy_p2 = np.zeros(dag_game.dag_structure_pl2.num_sequences)

        for _ in range(iterations):
            # Get strategies for both players
            strategy_p1 = player1.recommend()
            strategy_p2 = player2.recommend()

            # Update cumulative strategies
            cumulative_strategy_p1 += strategy_p1.treeplex_data
            cumulative_strategy_p2 += strategy_p2.treeplex_data

            # Compute payoff vector for each action
            reward_vector_p1 = np.zeros(dag_game.dag_structure_pl1.num_sequences)
            reward_vector_p2 = np.zeros(dag_game.dag_structure_pl2.num_sequences)
            for (seq_id1, seq_id2), reward in dag_game.leaves.items():
                reward_vector_p1[seq_id1] += reward * strategy_p2.treeplex_data[seq_id2]
                reward_vector_p2[seq_id2] -= reward * strategy_p1.treeplex_data[seq_id1]

            # Update regrets for both players
            player1.observe_rewards(reward_vector_p1, strategy_p1, inplace_rewards=True)
            player2.observe_rewards(reward_vector_p2, strategy_p2, inplace_rewards=True)

        # Normalize cumulative strategies to get average strategies
        avg_strategy_p1 = cumulative_strategy_p1 / iterations
        avg_strategy_p2 = cumulative_strategy_p2 / iterations

        return DagTreeplex(dag_game.dag_structure_pl1, avg_strategy_p1), DagTreeplex(dag_game.dag_structure_pl2, avg_strategy_p2)

def unit_test():
    """
    Try to solve Kuhn Poker
    """

    # Player 1 
    dag_structure_pl1 = DagStructure()
    dag_structure_pl1.add_infoset([0], 2, "J") # Check = 1, Bet = 2
    dag_structure_pl1.add_infoset([0], 2, "Q") # Check = 3, Bet = 4
    dag_structure_pl1.add_infoset([0], 2, "K") # Check = 5, Bet = 6

    dag_structure_pl1.add_infoset([1], 2, "J_C_b") # Fold = 7, Call = 8
    dag_structure_pl1.add_infoset([3], 2, "Q_C_b") # Fold = 9, Call = 10
    dag_structure_pl1.add_infoset([5], 2, "K_C_b") # Fold = 11, Call = 12

    # Player 2
    dag_structure_pl2 = DagStructure()
    dag_structure_pl2.add_infoset([0], 2, "j_C") # Check = 1, Bet = 2
    dag_structure_pl2.add_infoset([0], 2, "j_B") # Fold = 3, Call = 4
    dag_structure_pl2.add_infoset([0], 2, "q_C") # Check = 5, Bet = 6
    dag_structure_pl2.add_infoset([0], 2, "q_B") # Fold = 7, Call = 8
    dag_structure_pl2.add_infoset([0], 2, "k_C") # Check = 9, Bet = 10
    dag_structure_pl2.add_infoset([0], 2, "k_B") # Fold = 11, Call = 12

    # Construct leaves
    leaves = dict()
    
    # Add payoffs from P1 Betting and P2 Folding
    leaves[(2, 7)] = 1.0 / 6 # Jq
    leaves[(2, 11)] = 1.0 / 6 # Jk
    leaves[(4, 3)] = 1.0 / 6 # Qj
    leaves[(4, 11)] = 1.0 / 6 # Qk
    leaves[(6, 3)] = 1.0 / 6 # Kj
    leaves[(6, 7)] = 1.0 / 6 # Kq

    # Add payoffs from P1 Betting and P2 Calling
    leaves[(2, 8)] = -2.0 / 6 # Jq
    leaves[(2, 12)] = -2.0 / 6 # Jk
    leaves[(4, 4)] = 2.0 / 6 # Qj
    leaves[(4, 12)] = -2.0 / 6 # Qk
    leaves[(6, 4)] = 2.0 / 6 # Kj
    leaves[(6, 8)] = 2.0 / 6 # Kq

    # Add payoffs from P1 Checking and P2 Checking
    leaves[(1, 5)] = -1.0 / 6 # Jq
    leaves[(1, 9)] = -1.0 / 6 # Jk
    leaves[(3, 1)] =  1.0 / 6 # Qj
    leaves[(3, 9)] = -1.0 / 6 # Qk
    leaves[(5, 1)] = 1.0 / 6 # Kj
    leaves[(5, 5)] = 1.0 / 6 # Kq

    # Add payoffs from P1 checking, P2 betting, and P1 folding
    leaves[(7, 6)] = -1.0 / 6 # Jq
    leaves[(7, 10)] = -1.0 / 6 # Jk
    leaves[(9, 2)] =  -1.0 / 6 # Qj
    leaves[(9, 10)] = -1.0 / 6 # Qk
    leaves[(11, 2)] = -1.0 / 6 # Kj
    leaves[(11, 6)] = -1.0 / 6 # Kq
    
    # Add payoffs from P1 checking, P2 betting, and P1 calling
    leaves[(8, 6)] = -2.0 / 6 # Jq
    leaves[(8, 10)] = -2.0 / 6 # Jk
    leaves[(10, 2)] =  2.0 / 6 # Qj
    leaves[(10, 10)] = -2.0 / 6 # Qk
    leaves[(12, 2)] = 2.0 / 6 # Kj
    leaves[(12, 6)] = 2.0 / 6 # Kq

    dag_game = DagGame(dag_structure_pl1, dag_structure_pl2, leaves)
    strat1, strat2 = DagRegretMinimizer.solve_dag_game(dag_game, iterations=100000)
    print(strat1)
    print(strat2)
    game_val = dag_game.evaluate(strat1, strat2)
    print(game_val)
    print(dag_game.saddle_point_gap(strat1, strat2))

if __name__ == "__main__":
    unit_test()

# Example usage:
# Assuming DagStructure and RegretMatching are defined in their respective files.
# Example:
# dag = DagStructure(vertices=[1, 2, 3], edges=[(1, 2), (1, 3)])
# dag_minimizer = DagRegretMinimizer(dag)
# print(dag_minimizer.regret_matchers)
