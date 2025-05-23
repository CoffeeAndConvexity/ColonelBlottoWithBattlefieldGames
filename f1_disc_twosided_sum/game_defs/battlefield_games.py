from typing import List, Tuple
from game_defs.generalized_blotto import GeneralizedBBBlottoGame,BayesianBattlefieldGame
import copy
import numpy as np

class BlottoWithRaise(GeneralizedBBBlottoGame):
    def __init__(self,
                 num_battles: int,
                 num_soldiers: Tuple[int, int],
                 battlefields_worth: List[float] = None,
                 soft_victory: bool = False,
                 raise_multiplier : float = 2.0
                 ):
        """
        Initializes the Battlefield-Bayesian Blotto game with a specified number of players and battles.
        """

        if battlefields_worth is None:
            battlefields_worth = [1.0] * num_battles
        else:
            battlefields_worth = copy.deepcopy(battlefields_worth)

        self.num_battles = num_battles
        self.num_soldiers = num_soldiers
        
        bayesian_battlefield_games = []
        for battle_id in range(num_battles):
            bayesian_battlefield_game = BlottoWithRaise.generate_battlefield_game(num_soldiers[0],
                                                      num_soldiers[1],
                                                      battlefields_worth[battle_id],
                                                      soft_victory,
                                                      raise_multiplier)
            bayesian_battlefield_games.append(bayesian_battlefield_game)

        super().__init__(num_battles,
                       num_soldiers,
                       bayesian_battlefield_games)


    def generate_battlefield_game(max_soldiers_p1: int, 
                                  max_soldiers_p2: int, 
                                  battlefield_worth: float, 
                                  soft_victory: bool,
                                  raise_multiplier: float):
        """
        Action 1 is to keep and action 2 is to raise.
        """
        payoff_matrices = []
        for soldiers_p1 in range(max_soldiers_p1+1):
            payoff_matrices.append([])
            for soldiers_p2 in range(max_soldiers_p2+1):

                if soft_victory:
                    total_soldiers = soldiers_p1 + soldiers_p2
                    if total_soldiers == 0:
                        base_val = 0.0
                    else:
                        prob_p1_win = soldiers_p1 / total_soldiers
                        prob_p2_win = soldiers_p2 / total_soldiers
                        base_val = battlefield_worth * (prob_p1_win - prob_p2_win)
                else:
                    if soldiers_p1 > soldiers_p2:
                        base_val = battlefield_worth
                    elif soldiers_p1 < soldiers_p2:
                        base_val = -battlefield_worth
                    else:
                        base_val = 0.0

                # Fill in payoff matrix
                payoff_matrix = np.zeros((2, 2))

                # If neither player raises
                payoff_matrix[0, 0] = base_val * 1.0
                
                # If both players raise
                payoff_matrix[1, 1] = base_val * raise_multiplier * raise_multiplier

                # If player 1 raises and player 2 does not 
                payoff_matrix[1, 0] = base_val * raise_multiplier

                # If player 2 raises and player 1 does not
                payoff_matrix[0, 1] = base_val * raise_multiplier

                payoff_matrices[-1].append(payoff_matrix)

        return BayesianBattlefieldGame(max_soldiers_p1, 
                                       max_soldiers_p2,
                                       [2] * (max_soldiers_p1 + 1),
                                       [2] * (max_soldiers_p2 + 1),
                                       payoff_matrices)
''' 
# TODO: placeholder for now.
class BlottoWithSignalAndRaise(GeneralizedBBBlottoGame):
    def __init__(self,
                 num_battles: int,
                 num_soldiers: Tuple[int, int],
                 battlefields_worth: List[float] = None,
                 soft_victory: bool = False,
                 raise_multiplier : float = 2.0
                 ):
        """
        Initializes the Battlefield-Bayesian Blotto game with a specified number of players and battles.
        """

        if battlefields_worth is None:
            battlefields_worth = [1.0] * num_battles
        else:
            battlefields_worth = copy.deepcopy(battlefields_worth)

        self.num_battles = num_battles
        self.num_soldiers = num_soldiers
        
        bayesian_battlefield_games = []
        for battle_id in range(num_battles):
            bayesian_battlefield_game = BlottoWithRaise.generate_battlefield_game(num_soldiers[0],
                                                      num_soldiers[1],
                                                      battlefields_worth[battle_id],
                                                      soft_victory,
                                                      raise_multiplier)
            bayesian_battlefield_games.append(bayesian_battlefield_game)

        super().__init__(num_battles,
                       num_soldiers,
                       bayesian_battlefield_games)


    def generate_battlefield_game(max_soldiers_p1: int, 
                                  max_soldiers_p2: int, 
                                  battlefield_worth: float, 
                                  soft_victory: bool,
                                  raise_multiplier: float):
        """
        Action 1 is to keep and action 2 is to raise.
        """
        payoff_matrices = []
        for soldiers_p1 in range(max_soldiers_p1+1):
            payoff_matrices.append([])
            for soldiers_p2 in range(max_soldiers_p2+1):

                if soft_victory:
                    total_soldiers = soldiers_p1 + soldiers_p2
                    if total_soldiers == 0:
                        base_val = 0.0
                    else:
                        prob_p1_win = soldiers_p1 / total_soldiers
                        prob_p2_win = soldiers_p2 / total_soldiers
                        base_val = battlefield_worth * (prob_p1_win - prob_p2_win)
                else:
                    if soldiers_p1 > soldiers_p2:
                        base_val = battlefield_worth
                    elif soldiers_p1 < soldiers_p2:
                        base_val = -battlefield_worth
                    else:
                        base_val = 0.0

                # Fill in payoff matrix
                payoff_matrix = np.zeros((2, 2))

                # If neither player raises
                payoff_matrix[0, 0] = base_val * 1.0
                
                # If both players raise
                payoff_matrix[1, 1] = base_val * raise_multiplier * raise_multiplier

                # If player 1 raises and player 2 does not 
                payoff_matrix[1, 0] = base_val * raise_multiplier

                # If player 2 raises and player 1 does not
                payoff_matrix[0, 1] = base_val * raise_multiplier

                payoff_matrices[-1].append(payoff_matrix)

        return BayesianBattlefieldGame(max_soldiers_p1, 
                                       max_soldiers_p2,
                                       [2] * (max_soldiers_p1 + 1),
                                       [2] * (max_soldiers_p2 + 1),
                                       payoff_matrices)
'''