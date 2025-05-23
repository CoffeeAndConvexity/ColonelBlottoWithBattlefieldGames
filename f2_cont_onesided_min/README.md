# Main algorithm 
`subgrad_ascent_algo.py`: Implements Algorithm 1 from our paper: projected subgradient ascent on the soldiers simplex. Given parameterized subgame utilities, it returns the near-optimal soldier allocation of the maximizing player.

# Applications
## Randomly generated battlefield utilities:
In this experiment, the utility in every battlefield subgame is a randomly generated function of the number of soldiers that the maximizing player assigns to that battlefield. We consider affine and quadratic functions of the allocated soldiers.

## Security-inspired battlefield utilities: 
In this experiment, each battlefield features a two-player zero-sum security subgame in which Player 2 plays the defender’s role and Player 1 the attacker’s. For every battlefield, we employ a payoff matrix from *Krever, Noah, et al. GUARD: Constructing Realistic Two-Player Matrix and Security Games for Benchmarking Game-Theoretic Algorithms (2025)*, which generates datasets for realistic security-game instances.

# Running Experiment

From the `f2_cont_onesided_min` directory, you can run the experiment using the following command:
```
bash run_exps.sh
```

# Dependencies
- Python 
- `numpy`: for numerical computations. Install via:
```
pip install numpy
```


