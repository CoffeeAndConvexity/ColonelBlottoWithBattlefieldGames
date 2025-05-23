#pragma once
#include <stdio.h>
#include "battlefield_level_game.h"
BattlefieldLevelGame battlefield_with_raise(int num_soldiers_p1,
                                            int num_soldiers_p2,
                                            bool soft_victory,
                                            double raise_multiplier,
                                            double battlefield_worth)
{
    // TODO: handle memory leak.

    // Create a battlefield game with 5 soldiers for player 1 and 3 soldiers for player 2
    BattlefieldLevelGame battlefield(num_soldiers_p1,
                                     num_soldiers_p2);

    for (int num_soldiers_used_p1 = 0; num_soldiers_used_p1 <= num_soldiers_p1; num_soldiers_used_p1++)
    {
        for (int num_soldiers_used_p2 = 0; num_soldiers_used_p2 <= num_soldiers_p2; num_soldiers_used_p2++)
        {
            double showdown_payoff = 0.0;
            if (soft_victory == false)
            {
                if (num_soldiers_used_p1 > num_soldiers_used_p2)
                {
                    showdown_payoff = battlefield_worth;
                }
                else if (num_soldiers_used_p1 < num_soldiers_used_p2)
                {
                    showdown_payoff = -battlefield_worth;
                }
                else
                {
                    showdown_payoff = 0.0;
                }
            }
            else
            {
                int total_soldiers_sent = num_soldiers_used_p1 + num_soldiers_used_p2;
                if (total_soldiers_sent == 0)
                {
                    showdown_payoff = 0.0;
                }
                else
                {
                    double p1_win_prob = (double)num_soldiers_used_p1 / (double)total_soldiers_sent;
                    double p2_win_prob = (double)num_soldiers_used_p2 / (double)total_soldiers_sent;
                    showdown_payoff = battlefield_worth * (p1_win_prob - p2_win_prob);
                }
            }

            int num_actions_p1 = 2;
            int num_actions_p2 = 2;

            double *payoff_matrix = new double[num_actions_p1 * num_actions_p2];
            payoff_matrix[0] = showdown_payoff;                                       // P1: 0, P2: 0
            payoff_matrix[1] = showdown_payoff * raise_multiplier;                    // P1: 0, P2: 1
            payoff_matrix[2] = showdown_payoff * raise_multiplier;                    // P1: 1, P2: 0
            payoff_matrix[3] = showdown_payoff * raise_multiplier * raise_multiplier; // P1: 1, P2: 1

            battlefield.set_payoff_matrix(num_soldiers_used_p1,
                                          num_soldiers_used_p2,
                                          num_actions_p1,
                                          num_actions_p2,
                                          payoff_matrix);
        }
    }

    return battlefield;
}
