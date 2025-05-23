import numpy as np

class RegretMatching:
    def __init__(self, n):
        """
        Initialize the RegretMatching class.

        Parameters:
            n (int): Number of dimensions (actions).
        """
        self.n = n
        self.regrets = np.zeros(n)
        self.last_strategy = None

    def update_regrets(self, rewards, last_strategy=None):
        """
        Update the regrets based on the given rewards or losses.

        Parameters:
            rewards (list or np.ndarray): A list or array of rewards for each action.
        """
        if last_strategy is None:
            last_strategy = self.last_strategy

        if len(rewards) != self.n:
            raise ValueError("Length of rewards must match the number of dimensions (n).")

        # Update regrets based on the difference between the rewards and the last strategy
        reward_obtained = np.dot(rewards, last_strategy)
        self.regrets = self.regrets + (rewards - reward_obtained)

    def recommend(self):
        """
        Generate a recommendation (probability distribution) based on the current regrets.

        Returns:
            np.ndarray: A probability distribution over the actions.
        """
        # Ensure regrets are non-negative
        positive_regrets = np.maximum(self.regrets, 0)

        # Compute the sum of positive regrets
        total_positive_regret = np.sum(positive_regrets)

        # If total regret is zero, return a uniform distribution
        # Otherwise, normalize the positive regrets to form a probability distribution
        if total_positive_regret == 0:
            probabilities = np.ones(self.n) / self.n
        else:
            probabilities = positive_regrets / total_positive_regret

        # Store the last strategy for updating regrets in the future
        self.last_strategy = probabilities

        return probabilities
    
    def solve_matrix_game(matrix, iterations=10000):
        """
        Solve a matrix game using self-play with the RegretMatching class.

        Parameters:
            matrix (np.ndarray): A 2D array representing the payoff matrix for player 1.
            iterations (int): Number of iterations for self-play.

        Returns:
            tuple: A tuple containing the average strategy for player 1 and player 2.
        """
        n_actions_p1, n_actions_p2 = matrix.shape

        # Initialize regret matchers for both players
        player1 = RegretMatching(n_actions_p1)
        player2 = RegretMatching(n_actions_p2)

        # Initialize cumulative strategies
        cumulative_strategy_p1 = np.zeros(n_actions_p1)
        cumulative_strategy_p2 = np.zeros(n_actions_p2)

        for _ in range(iterations):
            # Get strategies for both players
            strategy_p1 = player1.recommend()
            strategy_p2 = player2.recommend()

            # Update cumulative strategies
            cumulative_strategy_p1 += strategy_p1
            cumulative_strategy_p2 += strategy_p2

            # Compute expected payoffs for each action
            payoff_p1 = matrix @ strategy_p2
            payoff_p2 = -matrix.T @ strategy_p1

            # Update regrets for both players
            player1.update_regrets(payoff_p1, strategy_p1)
            player2.update_regrets(payoff_p2, strategy_p2)

        # Normalize cumulative strategies to get average strategies
        avg_strategy_p1 = cumulative_strategy_p1 / iterations
        avg_strategy_p2 = cumulative_strategy_p2 / iterations

        return avg_strategy_p1, avg_strategy_p2

