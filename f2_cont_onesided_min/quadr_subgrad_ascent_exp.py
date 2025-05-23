import gurobipy as gp
from pathlib import Path
import sys
import numpy as np
from subgrad_ascent_algo import project_onto_simplex
from tqdm import tqdm
import pandas as pd

# CONSTANTS
ETA0_DEFAULT = 0.001
MAX_ITERS_DEFAULT = 10000
EPS_DEFAULT = 1e-2


def compute_nash_subgradient_quadratic(U0, U1, U2, x):
    m, n = U0.shape
    # build the payoff matrix at x
    Ux = U0 + x * U1 + (x**2) * U2

    model = gp.Model()
    model.setParam("OutputFlag", 0)

    # variables
    p = model.addVars(m, lb=0, ub=1)
    v = model.addVar(lb=-gp.GRB.INFINITY, name="v")

    # BR constraints
    for j in range(n):
        model.addConstr(
            gp.quicksum(Ux[i, j] * p[i] for i in range(m)) >= v,
            name=f"br_{j}"
        )
    model.addConstr(gp.quicksum(p[i] for i in range(m)) == 1)

    model.setObjective(v, gp.GRB.MAXIMIZE)
    model.optimize()

    v_opt = v.X

    # recover q* from duals
    duals = np.array([abs(model.getConstrByName(f"br_{j}").Pi) for j in range(n)])
    q_opt = duals / duals.sum()
    p_opt = np.array([p[i].X for i in range(m)])

    # subgradient: derivative of x*Ux*q wrt x
    term1 = p_opt @ U1 @ q_opt
    term2 = 2 * x * (p_opt @ U2 @ q_opt)
    dv_dx = term1 + term2

    return v_opt, dv_dx

    
def projected_gradient_ascent_quadratic(U0_list, U1_list, U2_list, x0, N, eta0, max_iters, eps, window=10):
    k = len(U0_list)
    x = np.array(x0, dtype=float)
    history, times = [], []
    import time
    start = time.time()

    for t in tqdm(range(max_iters)):
        v_vals, grads = [], []
        for i in range(k):
            v_i, dv_i = compute_nash_subgradient_quadratic(
                U0_list[i], U1_list[i], U2_list[i], x[i]
            )
            v_vals.append(v_i)
            grads.append(dv_i)

        # active battlefiled and record
        i_star = int(np.argmin(v_vals))
        history.append(v_vals[i_star])
        times.append(time.time() - start)

        # early stopping check
        if t >= window:
            recent = history[-window:]
            if max(recent) - min(recent) < eps:
                print(f"Converged at iter {t} (window range {max(recent)-min(recent):.2e} < {eps})")
                break
        
        eta_t = eta0 / np.sqrt(t + 1)  

        g = np.zeros(k)
        g[i_star] = grads[i_star]
        x = project_onto_simplex(x + eta_t * g, N)

    return x, history, times


def grad_ascent_quadr(subgame_size, seed, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random payoff matrices
    rng = np.random.default_rng(seed)
    k, N = 5, 20
    U0_list = [
        rng.uniform(-100, 100, (subgame_size, subgame_size))
        for _ in range(k)
    ]
    U1_list = [
        rng.uniform(0, 100, (subgame_size, subgame_size))
        for _ in range(k)
    ]
    U2_list = [
        rng.uniform(0, 100, (subgame_size, subgame_size))
        for _ in range(k)
    ]
    x_init = np.ones(k) * (N / k)

    # Run the ascent (with diminishing step‐size, early‐stop)
    x_opt, history, times = projected_gradient_ascent_quadratic(
        U0_list, U1_list, U2_list, x_init, N,
        eta0=ETA0_DEFAULT,
        max_iters=MAX_ITERS_DEFAULT,
        window=10,
        eps=EPS_DEFAULT
    )

    # Save per‐iteration history
    df_hist = pd.DataFrame({
        "iteration": np.arange(len(history)),
        "time":       times,
        "value":      history
    })
    hist_fn = output_dir / f"quadr_ascent_size{subgame_size}_seed{seed}.csv"
    df_hist.to_csv(hist_fn, index=False)

    # Save final x_opt
    df_xopt = pd.DataFrame([x_opt], columns=[f"x_{i}" for i in range(len(x_opt))])
    xopt_fn = output_dir / f"x_opt_size{subgame_size}_seed{seed}.csv"
    df_xopt.to_csv(xopt_fn, index=False)

    

if __name__== "__main__":
    # run python from bash with following parameters 
    try:
        subgame_size = int(sys.argv[1])
        seed= int(sys.argv[2])
        output_dir = Path(sys.argv[3]).resolve()
        grad_ascent_quadr(subgame_size, seed,  output_dir)
    except KeyboardInterrupt:
        sys.exit(130)
    