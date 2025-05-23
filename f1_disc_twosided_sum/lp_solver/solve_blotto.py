import cvxpy as cp
import numpy as np
from game_defs.generalized_blotto import GeneralizedBBBlottoGame
from online_learning.dag_regret_minimizer import DagGame
import gurobipy as gp

class LpSolver(object):
    def __init__(self, game: DagGame):
        self.game = game

    def solve_gurobi(self):
        game = self.game
        m= gp.Model("blotto")

        dag_p1 = game.dag_structure_pl1
        dag_p2 = game.dag_structure_pl2

        infoset_vals_var_p1 = m.addVars(dag_p1.num_infosets, name="infoset_vals_p1", lb=-gp.GRB.INFINITY)
        sequence_form_var_p2 = m.addVars(dag_p2.num_sequences, name="sequence_form_var_p2")
        # sequence_form_var_p2 = cp.Variable(dag_p2.num_sequences, nonneg=True)

        # Filter leaves based on sequence of p1
        leaves_for_seq_p1 = [[] for i in range(dag_p1.num_sequences)] 
        for seq_pair, val in game.leaves.items():
            seq_p1, seq_p2 = seq_pair
            leaves_for_seq_p1[seq_p1].append((seq_p2, val))

        # ==============================================================================
        # STEP 1)
        # Create dual first, the variables are the potentials for each infoset.
        # it is assumed that this is for the *min* player, so the dual would be
        # to *max*. The min player is assumed to be player 1. 
        # Make sure that the signs of the payoffs are correct!!!
        constrs = []
        # Get infoset infoset par->child mapping
        for infoset_id in range(dag_p1.num_infosets):
            start_seq_id = dag_p1.infoset_start_seq_id[infoset_id]
            num_actions = dag_p1.infoset_num_actions[infoset_id]
            for child_seq_id in range(start_seq_id, start_seq_id + num_actions):
                children_infosets = dag_p1.seq_id_child_infoset_id[child_seq_id]
                children_infoset_vals_var = [infoset_vals_var_p1[child_infoset_id] for child_infoset_id in children_infosets]

                # Payoffs for p1 that are from future infosets                
                future_payoff = gp.quicksum(children_infoset_vals_var)

                # Payoffs for p1 that are from immediate actions, assuming p1 played to be here
                immediate_payoff = gp.quicksum([sequence_form_var_p2[seq_p2] * payoff for seq_p2, payoff in leaves_for_seq_p1[child_seq_id]])

                m.addConstr(infoset_vals_var_p1[infoset_id] >= future_payoff + immediate_payoff)

        # STEP 2)
        # sequence form constraints for p2
        m.addConstr(sequence_form_var_p2[0] == 1.0)
        for infoset_id in range(dag_p2.num_infosets):
            start_seq_id = dag_p2.infoset_start_seq_id[infoset_id]
            num_actions = dag_p2.infoset_num_actions[infoset_id]
            parent_seq_ids = dag_p2.infoset_parent_seq_id[infoset_id]

            # Get the mass of the parent sequences
            parent_mass = gp.quicksum([sequence_form_var_p2[seq_id] for seq_id in parent_seq_ids])

            # Get child mass
            # child_mass = # gp.quicksum(sequence_form_var_p2[start_seq_id: start_seq_id + num_actions])
            child_mass = gp.quicksum(sequence_form_var_p2[j] for j in range(start_seq_id, start_seq_id + num_actions))

            m.addConstr(child_mass == parent_mass)

        # Objective
        # m.addConstrs(constrs, '')
        m.Params.Method = 1 # DUAL SIMPLEX
        m.setObjective(sum(infoset_vals_var_p1[infoset_id] for infoset_id in dag_p1.seq_id_child_infoset_id[0]), gp.GRB.MINIMIZE)
        m.optimize()

        #obj = cp.sum([infoset_vals_var_p1[infoset_id] for infoset_id in dag_p1.seq_id_child_infoset_id[0]])
        #problem = cp.Problem(cp.Minimize(obj), constrs)
        #problem.solve(solver= cp.GUROBI, verbose=True)
        #print(problem.value)


    def solve(self):
        game = self.game

        dag_p1 = game.dag_structure_pl1
        dag_p2 = game.dag_structure_pl2

        infoset_vals_var_p1 = cp.Variable(dag_p1.num_infosets)
        sequence_form_var_p2 = cp.Variable(dag_p2.num_sequences, nonneg=True)

        # Filter leaves based on sequence of p1
        leaves_for_seq_p1 = [[] for i in range(dag_p1.num_sequences)] 
        for seq_pair, val in game.leaves.items():
            seq_p1, seq_p2 = seq_pair
            leaves_for_seq_p1[seq_p1].append((seq_p2, val))

        # ==============================================================================
        # STEP 1)
        # Create dual first, the variables are the potentials for each infoset.
        # it is assumed that this is for the *min* player, so the dual would be
        # to *max*. The min player is assumed to be player 1. 
        # Make sure that the signs of the payoffs are correct!!!
        constrs = []
        # Get infoset infoset par->child mapping
        for infoset_id in range(dag_p1.num_infosets):
            start_seq_id = dag_p1.infoset_start_seq_id[infoset_id]
            num_actions = dag_p1.infoset_num_actions[infoset_id]
            for child_seq_id in range(start_seq_id, start_seq_id + num_actions):
                children_infosets = dag_p1.seq_id_child_infoset_id[child_seq_id]
                children_infoset_vals_var = [infoset_vals_var_p1[child_infoset_id] for child_infoset_id in children_infosets]

                # Payoffs for p1 that are from future infosets                
                future_payoff = cp.sum(children_infoset_vals_var)

                # Payoffs for p1 that are from immediate actions, assuming p1 played to be here
                immediate_payoff = cp.sum([sequence_form_var_p2[seq_p2] * payoff for seq_p2, payoff in leaves_for_seq_p1[child_seq_id]])

                constrs.append(infoset_vals_var_p1[infoset_id] >= future_payoff + immediate_payoff)

        # STEP 2)
        # sequence form constraints for p2
        constrs.append(sequence_form_var_p2[0] == 1.0)
        for infoset_id in range(dag_p2.num_infosets):
            start_seq_id = dag_p2.infoset_start_seq_id[infoset_id]
            num_actions = dag_p2.infoset_num_actions[infoset_id]
            parent_seq_ids = dag_p2.infoset_parent_seq_id[infoset_id]

            # Get the mass of the parent sequences
            parent_mass = cp.sum([sequence_form_var_p2[seq_id] for seq_id in parent_seq_ids])

            # Get child mass
            child_mass = cp.sum(sequence_form_var_p2[start_seq_id: start_seq_id + num_actions])

            constrs.append(child_mass == parent_mass)

        # Objective
        obj = cp.sum([infoset_vals_var_p1[infoset_id] for infoset_id in dag_p1.seq_id_child_infoset_id[0]])
        problem = cp.Problem(cp.Minimize(obj), constrs)
        problem.solve(solver= cp.GUROBI, verbose=True)
        print(problem.value)