#pragma once

#include <iostream>
#include <stdio.h>

#include <random>
#include <cstdlib>
#include <vector>
#include <map>

class BattlefieldLevelGame
{
public:
    int max_soldiers_p1;
    int max_soldiers_p2;

    int *num_actions_p1;
    int *num_actions_p2;

    double **payoff_matrices;

    BattlefieldLevelGame(int max_soldiers_p1,
                         int max_soldiers_p2);
    void set_payoff_matrix(int soldiers_p1, int soldiers_p2, int matrix_actions_p1, int matrix_actions_p2, double *payoff_matrix);
    double get_payoffs(int soldiers_p1, int soldiers_p2, int action_p1, int action_p2) const;
};
