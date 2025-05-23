#include <iostream>
#include <stdio.h>

#include "regret_minimizer_efce_flattened_base.h"
#include "leaf.h"
#include "battlefield_level_game.h"
#include <random>
#include <cstdlib>
#include <vector>
#include <map>

class Blotto
{
public:
    int num_battlefields;
    int max_soldiers_p1;
    int max_soldiers_p2;

    BattlefieldLevelGame *battlefield_games;

    Blotto(int num_battlefields, int max_soldiers_p1, int max_soldiers_p2, BattlefieldLevelGame *battlefield_games, RegretMinimizerEfceFlattenedBase *rm_p1, RegretMinimizerEfceFlattenedBase *rm_p2);
    double get_payoffs(int battlefield_id, int soldiers_p1, int soldiers_p2, int action_p1, int action_p2) const;
    double evaluate(double *strat_p1, double *strat_p2);
    void solve(unsigned int num_iterations, bool verbose = false);
    void get_loss_vector_pl1(double *strat_pl2, double *storage);
    void get_loss_vector_pl2(double *strat_pl1, double *storage);

    double *get_solution_p1();
    double *get_solution_p2();

    // private:
    // Storage space for solution
    double *ret_strat_p1;
    double *ret_strat_p2;

    // Get starting sequences for each infoset
    std::map<std::pair<int, int>, int> bf_and_starting_soldiers_to_start_seq_id_p1; // [battlefield_id, num soliders left] -> starting sequence id
    std::map<std::pair<int, int>, int> bf_and_starting_soldiers_to_start_seq_id_p2;
    std::map<std::pair<int, int>, int> bf_and_used_soldiers_to_start_seq_id_p1; // [battlefield_id, num soliders used] -> starting sequence id
    std::map<std::pair<int, int>, int> bf_and_used_soldiers_to_start_seq_id_p2;

    // Regret minimizers
    RegretMinimizerEfceFlattenedBase *rm_p1;
    RegretMinimizerEfceFlattenedBase *rm_p2;
    std::vector<Leaf> leaves;

    void construct_xi_polytope(int player_id, // 0 or 1
                               RegretMinimizerEfceFlattenedBase *rm,
                               std::map<std::pair<int, int>, int> *infosets_start_seq,
                               std::map<std::pair<int, int>, int> *dummy_start_seq);
    void construct_leaves();
};