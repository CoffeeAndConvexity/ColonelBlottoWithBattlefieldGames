from game_defs.battlefield_games import BlottoWithRaise
import unittest
import numpy as np
from online_learning.dag_regret_minimizer import DagRegretMinimizer
from lp_solver.solve_blotto import LpSolver

def unit_test():
    
    # CASE 1
    num_battlefields = 3
    battlefield_worth = (np.array(list(range(num_battlefields)))+ 1)/(num_battlefields * (1.0 + num_battlefields)/2) 
    game = BlottoWithRaise(3, (5, 3), battlefield_worth, soft_victory=True, raise_multiplier=2.0)
    #strat_p1, strat_p2 = DagRegretMinimizer.solve_dag_game(game, iterations=10000)

    # CASE 2
    #num_battlefields = 30
    #battlefield_worth = (np.array(list(range(num_battlefields)))+ 1)/(num_battlefields * (1.0 + num_battlefields)/2) 
    #game = BlottoWithRaise(num_battlefields, (100, 50), battlefield_worth, soft_victory=True, raise_multiplier=2.0)


    # Case 3 Problem with numerical issues?
    #num_battlefields = 35
    #battlefield_worth = (np.array(list(range(num_battlefields)))+ 1)/(num_battlefields * (1.0 + num_battlefields)/2) 
    #game = BlottoWithRaise(num_battlefields, (125, 70), battlefield_worth, soft_victory=True, raise_multiplier=2.0)

    # Case 4 Correct ans but 5000s
    # num_battlefields = 40
    # battlefield_worth = (np.array(list(range(num_battlefields)))+ 1)/(num_battlefields * (1.0 + num_battlefields)/2) 
    # game = BlottoWithRaise(num_battlefields, (150, 100), battlefield_worth, soft_victory=True, raise_multiplier=2.0)

    # Case 5 Numerical problems.
    # num_battlefields = 50
    # battlefield_worth = (np.array(list(range(num_battlefields)))+ 1)/(num_battlefields * (1.0 + num_battlefields)/2) 
    # game = BlottoWithRaise(num_battlefields, (200, 100), battlefield_worth, soft_victory=True, raise_multiplier=2.0)

    lp_solver = LpSolver(game)
    lp_solver.solve_gurobi()


if __name__ == '__main__':
    unit_test()
