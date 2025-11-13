# Colonel Blotto with Battlefield Games
This repository implements the experiments we perform in our paper. 

## Discrete two-sided with sum aggregator
In this experiment, we construct a two-sided discrete two-level Blotto game under the sum aggregator. We consider (i) the LP-based approach, and (ii) our approach based on online learning. 
### LP-based approach
Run ./unit_tests/test_lp_solver.py (with the appropriate lines commented).
### Online Learning method
For online learning method, cd to ./fast and run make. 

Uncomment (and comment) the relevant lines, and the executable generated should be blotto_basic or blotto_alt. 
The former is for the basic (slow) version, and blotto_alt should be the faster version.

To change the size of the game and/or the method used, modify blotto_basic.cpp or blotto_basic_speedup.cpp respectively.

**NOTE 1:** this is a minimum working example of our code that may be used to reproduce our experimental results for discrete two-sided blotto with additive payoffs.

We have not put in much effort into making the code maintainable for this reason.

**NOTE 2:** This code was an attempt at a faithful reimplementation of Farina et. al (2019) where the scaled extension was presented in the context of EFCEs. Hence, part of the C++ implementation follows their convention. We are merely adopting it here for our Blotto setting.

## Continuous one-sided with min-aggregator
In this experiment, we evaluate our subgradient ascent algorithm to solve a one-sided continuous two-level Blotto game under the min aggregator. We consider (1) randomly generated and (2) security-inspired battlefield utilities for the subgames. 
### Main algorithm 
`subgrad_ascent_algo.py`: Implements Algorithm 1 from our paper: projected subgradient ascent on the soldiers simplex. Given parameterized subgame utilities, it returns the near-optimal soldier allocation of the maximizing player.

From the `f2_cont_onesided_min` directory, you can run the experiment using the following command:
```
bash run_exps.sh
```

### Randomly generated battlefield utilities:
In this experiment, the utility in every battlefield subgame is a randomly generated function of the number of soldiers that the maximizing player assigns to that battlefield. We consider affine and quadratic functions of the allocated soldiers.

### Security-inspired battlefield utilities: 
In this experiment, each battlefield features a two-player zero-sum security subgame in which Player 2 plays the defender’s role and Player 1 the attacker’s. For every battlefield, we employ a payoff matrix from *Krever, Noah, et al. GUARD: Constructing Realistic Two-Player Matrix and Security Games for Benchmarking Game-Theoretic Algorithms (2025)*, which generates datasets for realistic security-game instances.


# Dependencies
- Python
- C++
- `numpy`: for numerical computations

# Citation
If you use this repository, please cite our paper:

Afiouni, S., Cerny, J., Ling, C. K., & Kroer, C. (2025). Colonel Blotto with Battlefield Games. arXiv preprint arXiv:2511.06518.
```
@article{afiouni2025colonel,
  title={Colonel Blotto with Battlefield Games},
  author={Afiouni, Salam and Cerny, Jakub and Ling, Chun Kai and Kroer, Christian},
  journal={arXiv preprint arXiv:2511.06518},
  year={2025}
}
```

Full paper with the entire appendix available on [arxiv](https://arxiv.org/abs/2511.06518).

