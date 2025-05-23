from game_defs.basic_blotto import BlottoGame
from online_learning.dag_regret_minimizer import DagRegretMinimizer
import unittest
import numpy as np

class TestBlottoGame(unittest.TestCase):
    def test_sanity(self):
        print('test sanity')
        num_battles = 5
        num_soldiers = (3, 0)
        game = BlottoGame(num_battles, num_soldiers, [1.0, 2.0, 3.0, 4.0, 5.0])

        strat_p1, strat_p2 = DagRegretMinimizer.solve_dag_game(game, iterations=1000)
        game_value = game.evaluate(strat_p1, strat_p2)
        print('Game value', game_value)
        assert np.isclose(game_value, 12.0, atol=1e-1), f"Game value not close, {game_value}"

        saddle_point_gap = game.saddle_point_gap(strat_p1, strat_p2)
        print('saddle point gap', saddle_point_gap)
        assert saddle_point_gap < 1e-1, f"Saddle point gap too large: {saddle_point_gap}"

    def test_trivial(self):
        print('test trivial')
        num_battles = 5
        num_soldiers = (10, 1)
        game = BlottoGame(num_battles, num_soldiers, [1.0, 2.0, 3.0, 4.0,5.0])

        strat_p1, strat_p2 = DagRegretMinimizer.solve_dag_game(game, iterations=1000)
        print('Game value', game.evaluate(strat_p1, strat_p2))
        
        saddle_point_gap = game.saddle_point_gap(strat_p1, strat_p2)
        print('saddle point gap', saddle_point_gap)
        assert saddle_point_gap < 1e-1, f"Saddle point gap too large: {saddle_point_gap}"
        
    def test_trivial2(self):
        print('test trivial2')
        num_battles = 3
        num_soldiers = (3, 1)
        game = BlottoGame(num_battles, num_soldiers, [5.0, 5.0, 5.0])

        strat_p1, strat_p2 = DagRegretMinimizer.solve_dag_game(game, iterations=2000)
        game_value = game.evaluate(strat_p1, strat_p2)
        print('Game value', game_value)
        assert np.isclose(game_value, 10.0, atol=1e-1), f"Game value not close, {game_value}"

        saddle_point_gap = game.saddle_point_gap(strat_p1, strat_p2)
        print('saddle point gap', saddle_point_gap)
        assert saddle_point_gap < 1e-1, f"Saddle point gap too large: {saddle_point_gap}"

    def test_solver_tiny(self):
        
        print('Testing tiny')
        num_battles = 2
        num_soldiers = (3, 1)
        game = BlottoGame(num_battles, num_soldiers)

        strat_p1, strat_p2 = DagRegretMinimizer.solve_dag_game(game, iterations=1000)
        print('Game value', game.evaluate(strat_p1, strat_p2))
        game_value = game.evaluate(strat_p1, strat_p2)
        print('Game value', game_value)
        assert np.isclose(game_value, 1.5, atol=1e-1), f"Game value not close, {game_value}"
        
        saddle_point_gap = game.saddle_point_gap(strat_p1, strat_p2)
        print('saddle point gap', saddle_point_gap)
        assert saddle_point_gap < 1e-1, f"Saddle point gap too large: {saddle_point_gap}"

    def test_solver_medium(self):
        print('Testing medium size')
        num_battles = 5
        num_soldiers = (5, 10)
        game = BlottoGame(num_battles, num_soldiers, [1.0, 2.0, 3.0, 4.0, 5.0])

        strat_p1, strat_p2 = DagRegretMinimizer.solve_dag_game(game, iterations=50000)
        game_value = game.evaluate(strat_p1, strat_p2)
        print('Game value', game_value)
        assert game_value < 0.0, f"Game value should be negative, {game_value}"
        
        saddle_point_gap = game.saddle_point_gap(strat_p1, strat_p2)
        print('saddle point gap', saddle_point_gap)
        assert saddle_point_gap < 1e-1, f"Saddle point gap too large: {saddle_point_gap}"

if __name__ == "__main__":
    unittest.main()