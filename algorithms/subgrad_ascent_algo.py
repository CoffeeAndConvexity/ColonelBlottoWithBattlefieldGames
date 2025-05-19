import gurobipy as gp
import time
import numpy as np
import time
import matplotlib.pyplot as plt


# Solve the zero‐sum subgame and compute v(x) plus its subgradient
def compute_nash_subgradient(U0, U1, x):

    m, n = U0.shape

    # game matrix
    Ux = U0 + x * U1 
    
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
    
    # compute subgradient dv/dx = p^T U1 q (from duals)
    dq_dx = np.array([abs(model.getConstrByName(f"br_{j}").Pi) for j in range(n)])
    q_opt = dq_dx / np.sum(dq_dx)  # Normalize to sum to 1
    p_opt = np.array([p[i].X for i in range(m)])
    
    # nash subgradient
    dv_dx = p_opt @ U1 @ q_opt
    
    return v_opt, dv_dx

# Projection onto the simplex { x ≥ 0 : ∑ x_i = z } (Duchi et al. ICML ’08)
def project_onto_simplex(v, z):
    k = len(v)
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.where(u > (sv - z) / (np.arange(k) + 1))[0][-1]
    theta = (sv[rho] - z) / (rho + 1)
    return np.maximum(v - theta, 0)


def projected_gradient_ascent(U0_list, U1_list, x0, N, step_size, max_iters):
    k = len(U0_list) # number of battlefields
    x = np.array(x0, dtype=float)
    history, times = [], []
    start = time.time()

    for t in range(max_iters):
        # compute v_i and subgradient for each battlefield
        v_vals = []
        grads  = []
        for i in range(k):
            v_i, dv_dx_i = compute_nash_subgradient(U0_list[i], U1_list[i], x[i])
            v_vals.append(v_i)
            grads.append(dv_dx_i)

        # active battlefield
        i_star = int(np.argmin(v_vals))
        history.append(v_vals[i_star])
        times.append(time.time() - start)

        # subgradient only in the i_star coordinate
        g = np.zeros(k)
        g[i_star] = grads[i_star]

        # gradient‐ascent + projection
        x = project_onto_simplex(x + step_size * g, N)

    return x, history, times


# Main script: generate random games, run ascent, and plot
if __name__ == "__main__":
    # problem dimensions
    k = 2           # number of battlefields
    m, n = 10, 10   # payoff matrix size
    N = 3           # total soldiers

    # reproducible random payoff matrices
    np.random.seed(0)
    U0_list = [np.random.uniform(-5, 5, (m, n)) for _ in range(k)]
    U1_list = [np.random.uniform(0, 3,  (m, n)) for _ in range(k)]

    # initial allocation (uniform)
    x_init = np.ones(k)
    x_init = x_init / x_init.sum() * N

    # run projected subgradient ascent
    x_opt, history, times = projected_gradient_ascent(
        U0_list, U1_list, x_init, N,
        step_size=0.01, max_iters=30
    )

    print(f"\nOptimal allocation: {x_opt}")
    print(f"Optimal min Nash value: {history[-1]:.4f}\n")

    # plot objective vs. iteration 
    plt.figure(figsize=(6,4))
    plt.plot(history, linewidth=1.5)
    plt.xlabel("Iteration $t$")
    plt.ylabel("Objective $f(x^{(t)})$")
    plt.title("Convergence of Projected Gradient Ascent")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot objective vs. time
    plt.figure(figsize=(6,4))
    plt.plot(times, history, linewidth=1.5)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Objective $f(x^{(t)})$")
    plt.title("f vs. Real Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()