from game_defs.battlefield_games import BlottoWithRaise
import unittest
import numpy as np
from online_learning.dag_regret_minimizer import DagRegretMinimizer

def unit_test():
    game = BlottoWithRaise(3, (5, 3), [1.0, 2.0, 3.0], soft_victory=True, raise_multiplier=2.0)
    strat_p1, strat_p2 = DagRegretMinimizer.solve_dag_game(game, iterations=10000)
    val = game.evaluate(strat_p1, strat_p2)
    print(val)

if __name__ == '__main__':
    unit_test()