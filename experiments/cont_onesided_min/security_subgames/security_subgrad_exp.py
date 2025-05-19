import gurobipy as gp
from pathlib import Path
import time
import csv
import numpy as np
from algorithms.subgrad_ascent_algo import project_onto_simplex
from tqdm import tqdm
import pandas as pd
import pickle

# CONSTANTS
ETA0_DEFAULT = 0.01
MAX_ITERS_DEFAULT = 1000

def compute_nash_subgradient(U0, C, x):
    m, n = U0.shape

    # game matrix
    Ux = U0 + np.log(x+1) * C
    
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    
    p = model.addVars(m, lb=0, ub=1)
    v = model.addVar(name="v", lb=-float("inf"))
    
    # br constraints
    for j in range(n):
        model.addConstr(gp.quicksum(Ux[i, j] * p[i] for i in range(m)) >= v, name=f"br_{j}")
    
    model.addConstr(gp.quicksum(p[i] for i in range(m)) == 1)
    
    model.setObjective(v, gp.GRB.MAXIMIZE)
    
    model.optimize()
    
    v_opt = v.X
    
    # compute subgradient dv/dx = p^T U1 q 
    dq_dx = np.array([abs(model.getConstrByName(f"br_{j}").Pi) for j in range(n)])
    q_opt = dq_dx / np.sum(dq_dx)  
    p_opt = np.array([p[i].X for i in range(m)])
    
    # nash subgradient
    dv_dx = (1/(x+1)) * p_opt @ C @ q_opt
    
    return v_opt, dv_dx



def projected_gradient_ascent(
    U0_list, C_list, x0, N, 
    eta0, max_iters
):

    k = len(U0_list)
    x = np.array(x0, dtype=float)
    history, times = [], []
    start = time.time()

    for t in tqdm(range(max_iters)):
        # compute v_i and gradient for each battlefield
        v_vals, grads = [], []
        for i in range(k):
            v_i, dv_dx_i = compute_nash_subgradient(U0_list[i], C_list[i], x[i])
            v_vals.append(v_i)
            grads.append(dv_dx_i)

        # active battlefiled and record
        i_star = int(np.argmin(v_vals))
        history.append(v_vals[i_star])
        times.append(time.time() - start)

        eta_t = eta0 / np.sqrt(t + 1)  

        # asc step + projection
        g = np.zeros(k);  g[i_star] = grads[i_star]
        x = project_onto_simplex(x + eta_t * g, N)

    return x, history, times



def grad_ascent_runs_security(pickle_paths, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    k = 3
    N = 10
    seeds = [0, 1, 2]
    U0_list = []
    C_list = []
    for i, pkl in enumerate(pickle_paths):
        with open(pkl, "rb") as f:
            obj = pickle.load(f)
            defender_arr = np.array(obj['defender_utility_matrix'])            
            rng = np.random.default_rng(seeds[i])
            C = rng.uniform(0, 1, size=defender_arr.shape)
           
            C_list.append(C)
        U0_list.append(defender_arr)

    x_init = np.ones(k) * (N / k)

    # Run the ascent with diminishing step‐size, max_iters
    x_opt, history, times = projected_gradient_ascent(
        U0_list, C_list, x_init, N,
        eta0=ETA0_DEFAULT,
        max_iters=MAX_ITERS_DEFAULT
    )

    # Save per‐iteration history
    df_hist = pd.DataFrame({
        "iteration": np.arange(len(history)),
        "time":       times,
        "value":      history
    })
    hist_fn = output_dir / "linear_ascent.csv"
    df_hist.to_csv(hist_fn, index=False)

    # Save final x_opt
    df_xopt = pd.DataFrame([x_opt], columns=[f"x_{i}" for i in range(len(x_opt))])
    xopt_fn = output_dir / "x_opt_size.csv"
    df_xopt.to_csv(xopt_fn, index=False)




if __name__== "__main__":
    output_dir = Path("results/sec_grad_ascent")
    pickle_paths = [
        "data_security_subgames/small_isg_1.pkl",
        "data_security_subgames/small_isg_2.pkl",
        "data_security_subgames/small_isg_3.pkl"
    ]
    grad_ascent_runs_security(pickle_paths, output_dir)