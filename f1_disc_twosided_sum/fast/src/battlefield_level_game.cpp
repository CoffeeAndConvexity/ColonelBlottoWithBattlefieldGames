#include "battlefield_level_game.h"

BattlefieldLevelGame::BattlefieldLevelGame(int max_soldiers_p1,
                                           int max_soldiers_p2) : max_soldiers_p1(max_soldiers_p1),
                                                                  max_soldiers_p2(max_soldiers_p2)
{
    num_actions_p1 = new int[max_soldiers_p1 + 1];
    num_actions_p2 = new int[max_soldiers_p2 + 1];

    std::fill(num_actions_p1, num_actions_p1 + max_soldiers_p1 + 1, -1);
    std::fill(num_actions_p2, num_actions_p2 + max_soldiers_p2 + 1, -1);

    payoff_matrices = new double *[(max_soldiers_p1 + 1) * (max_soldiers_p2 + 1)];
    for (int i = 0; i <= max_soldiers_p1; i++)
    {
        for (int j = 0; j <= max_soldiers_p2; j++)
        {
            payoff_matrices[i * (max_soldiers_p2 + 1) + j] = NULL;
        }
    }
}

void BattlefieldLevelGame::set_payoff_matrix(int soldiers_p1,
                                             int soldiers_p2,
                                             int matrix_actions_p1,
                                             int matrix_actions_p2,
                                             double *payoff_matrix)
{
    assert(soldiers_p1 <= max_soldiers_p1 && soldiers_p2 <= max_soldiers_p2);
    assert(soldiers_p1 >= 0 && soldiers_p2 >= 0);

    if (num_actions_p1[soldiers_p1] == -1)
    {
        num_actions_p1[soldiers_p1] = matrix_actions_p1;
    }
    else
    {
        assert(num_actions_p1[soldiers_p1] == matrix_actions_p1);
    }

    if (num_actions_p2[soldiers_p2] == -1)
    {
        num_actions_p2[soldiers_p2] = matrix_actions_p2;
    }
    else
    {
        assert(num_actions_p2[soldiers_p2] == matrix_actions_p2);
    }

    payoff_matrices[soldiers_p1 * (max_soldiers_p2 + 1) + soldiers_p2] = payoff_matrix;
}

double BattlefieldLevelGame::get_payoffs(int soldiers_p1, int soldiers_p2, int action_p1, int action_p2) const
{
    assert(soldiers_p1 <= max_soldiers_p1 && soldiers_p2 <= max_soldiers_p2);
    assert(soldiers_p1 >= 0 && soldiers_p2 >= 0);
    assert(action_p1 >= 0 && action_p1 < num_actions_p1[soldiers_p1]);
    assert(action_p2 >= 0 && action_p2 < num_actions_p2[soldiers_p2]);

    double *payoff_matrix = payoff_matrices[soldiers_p1 * (max_soldiers_p2 + 1) + soldiers_p2];
    return payoff_matrix[action_p1 * num_actions_p2[soldiers_p2] + action_p2];
}
