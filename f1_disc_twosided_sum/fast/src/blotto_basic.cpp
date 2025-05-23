#include <iostream>
#include <stdio.h>

#include "regret_minimizer_efce_flattened_base.h"
#include "prmplus_efce_flattened.h"
#include "prm_efce_flattened.h"
#include "rmplus_efce_flattened.h"
#include "rm_efce_flattened.h"
#include <random>
#include <cstdlib>
#include "blotto.h"
#include <chrono>

#define CASE 5

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

int main()
{
#if CASE == 1
    // CASE 1 //
    const int num_soldiers_p1 = 5;
    const int num_soldiers_p2 = 3;
    const int num_battlefields = 3;
    double battlefield_worth[num_battlefields];
    double sum_battlefield_worth = (num_battlefields * (1.0 + num_battlefields)) / 2;
    for (int x = 0; x < num_battlefields; x++)
    {
        battlefield_worth[x] = (1.0 + (double)x) / sum_battlefield_worth;
    }
    const bool soft_victory = true;
    double raise_multiplier = 2.0;
#elif CASE == 2
    const int num_soldiers_p1 = 100;
    const int num_soldiers_p2 = 50;
    const int num_battlefields = 30;
    double battlefield_worth[num_battlefields];
    double sum_battlefield_worth = (num_battlefields * (1.0 + num_battlefields)) / 2;
    for (int x = 0; x < num_battlefields; x++)
    {
        battlefield_worth[x] = (1.0 + (double)x) / sum_battlefield_worth;
    }
    const bool soft_victory = true;
    double raise_multiplier = 2.0;
#elif CASE == 3
    const int num_soldiers_p1 = 125;
    const int num_soldiers_p2 = 70;
    const int num_battlefields = 35;
    double battlefield_worth[num_battlefields];
    double sum_battlefield_worth = (num_battlefields * (1.0 + num_battlefields)) / 2;
    for (int x = 0; x < num_battlefields; x++)
    {
        battlefield_worth[x] = (1.0 + (double)x) / sum_battlefield_worth;
    }
    const bool soft_victory = true;
    double raise_multiplier = 2.0;
#elif CASE == 4
    // CASE 3 //
    const int num_soldiers_p1 = 200;
    const int num_soldiers_p2 = 100;
    const int num_battlefields = 50;
    double battlefield_worth[num_battlefields];
    double sum_battlefield_worth = (num_battlefields * (1.0 + num_battlefields)) / 2;
    for (int x = 0; x < num_battlefields; x++)
    {
        battlefield_worth[x] = (1.0 + (double)x) / sum_battlefield_worth;
    }
    const bool soft_victory = true;
    double raise_multiplier = 2.0;

#elif CASE == 5
    // CASE 3 //
    const int num_soldiers_p1 = 500;
    const int num_soldiers_p2 = 200;
    const int num_battlefields = 100;
    double battlefield_worth[num_battlefields];
    double sum_battlefield_worth = (num_battlefields * (1.0 + num_battlefields)) / 2;
    for (int x = 0; x < num_battlefields; x++)
    {
        battlefield_worth[x] = (1.0 + (double)x) / sum_battlefield_worth;
    }
    const bool soft_victory = true;
    double raise_multiplier = 2.0;
#endif
    std::vector<BattlefieldLevelGame> battlefields_vec;

    for (int i = 0; i < num_battlefields; i++)
    {
        BattlefieldLevelGame battlefield = battlefield_with_raise(num_soldiers_p1,
                                                                  num_soldiers_p2,
                                                                  soft_victory,
                                                                  raise_multiplier,
                                                                  battlefield_worth[i]);
        battlefields_vec.push_back(battlefield);
    }

    // auto rm_p1 = new RmPlusEfceFlattened();
    // auto rm_p2 = new RmPlusEfceFlattened();

    // auto rm_p1 = new RmEfceFlattened();
    // auto rm_p2 = new RmEfceFlattened();

    // auto rm_p1 = new PrmPlusEfceFlattened();
    // auto rm_p2 = new PrmPlusEfceFlattened();

    auto rm_p1 = new PrmEfceFlattened();
    auto rm_p2 = new PrmEfceFlattened();

    Blotto blotto(num_battlefields, num_soldiers_p1, num_soldiers_p2, battlefields_vec.data(), rm_p1, rm_p2);
    std::cout << "Num sequences: " << blotto.rm_p1->size() << " " << blotto.rm_p2->size() << "\n";
    std::cout << "Solving...\n";
    auto start = std::chrono::high_resolution_clock::now();
    blotto.solve(100000);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "done\n";
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Solving time: " << elapsed.count() << " seconds\n";
    std::cout << "\n";

    double val = blotto.evaluate(blotto.get_solution_p1(), blotto.get_solution_p2());
    std::cout << val;
    return 0;
}