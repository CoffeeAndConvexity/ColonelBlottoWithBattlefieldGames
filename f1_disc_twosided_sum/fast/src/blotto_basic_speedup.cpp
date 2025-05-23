#include <iostream>
#include <stdio.h>

#include "regret_minimizer_efce_flattened_base.h"
#include "prmplus_efce_flattened.h"
#include "prm_efce_flattened.h"
#include "rmplus_efce_flattened.h"
#include "leaf.h"
#include <random>
#include <cstdlib>
#include "blotto_alt.h"
#include <chrono>
#include "battlefield_with_raise.hpp"

#define CASE 2

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

    auto rm_p1 = new PrmPlusEfceFlattened();
    auto rm_p2 = new PrmPlusEfceFlattened();

    // auto rm_p1 = new PrmEfceFlattened();
    //  auto rm_p2 = new PrmEfceFlattened();
    BlottoAlt blotto(num_battlefields, num_soldiers_p1, num_soldiers_p2, battlefields_vec.data(), rm_p1, rm_p2);
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