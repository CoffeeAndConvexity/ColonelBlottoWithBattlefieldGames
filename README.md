# ColonelBlottoWithBattlefieldGames
This repository implements a projected subgradient ascent algorithm to compute the optimal continuous soldier‐allocation strategy for the maximizing player in the one-sided, two-level Colonel Blotto game with the min payoff aggregator.  We demonstrate the efficacy of our approach on: (1) Randomly generated battlefield utilities (affine and quadratic functions of the allocated soldiers), and (2) Security-inspired battlefield utilities drawn from real-world datasets.


# Main algorithm 
`subgrad_ascent_algo.py`: Implements Algorithm 1 from our paper: projected subgradient ascent on the soldiers simplex.  Given parameterized subgame utilities, it returns the near-optimal soldier allocation of the maximizing player.


# Applications
## Randomly generated battlefiled utilities:
In this experiment, the utility in every battlefield subgame is a randomly generated function of the number of soldiers that the maximizing player asigns to that battlefiled. We consider affine and quadratic functions.

## Security-inspired battlefiled utilites: 
In this experiment, qach battlefield features a two-player zero-sum security subgame in which Player 2 plays the defender’s role and Player 1 the attacker’s. For every battlefield, we employ a payoff matrix from *Krever, Noah, et al. GUARD: Constructing Realistic Two-Player Matrix and Security Games for Benchmarking Game-Theoretic Algorithms (2025)*, which generates datasets for realistic security-game instances.

# Dependencies
- Python 
- `numpy`: for numerical computations. Install via:
```
pip install numpy
```


