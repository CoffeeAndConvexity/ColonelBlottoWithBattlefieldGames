#include "blotto_alt.h"

BlottoAlt::BlottoAlt(int num_battlefields,
                     int max_soldiers_p1,
                     int max_soldiers_p2,
                     BattlefieldLevelGame *battlefield_games,
                     RegretMinimizerEfceFlattenedBase *rm_p1,
                     RegretMinimizerEfceFlattenedBase *rm_p2) : num_battlefields(num_battlefields),
                                                                max_soldiers_p1(max_soldiers_p1),
                                                                max_soldiers_p2(max_soldiers_p2),
                                                                battlefield_games(battlefield_games),
                                                                rm_p1(rm_p1),
                                                                rm_p2(rm_p2)
{
    construct_xi_polytope(0, rm_p1, &bf_and_starting_soldiers_to_start_seq_id_p1, &bf_and_used_soldiers_to_start_seq_id_p1);
    construct_xi_polytope(1, rm_p2, &bf_and_starting_soldiers_to_start_seq_id_p2, &bf_and_used_soldiers_to_start_seq_id_p2);

    construct_leaves();
}

double BlottoAlt::get_payoffs(int battlefield_id, int soldiers_p1, int soldiers_p2, int action_p1, int action_p2) const
{
    return battlefield_games[battlefield_id].get_payoffs(soldiers_p1, soldiers_p2, action_p1, action_p2);
}

double *BlottoAlt::get_solution_p1()
{
    return ret_strat_p1;
}
double *BlottoAlt::get_solution_p2()
{
    return ret_strat_p2;
}

double BlottoAlt::evaluate(double *strat_p1, double *strat_p2)
{
    double payoff = 0.0;
    for (int i = 0; i < num_battlefields; i++)
    {
        for (int soldiers_p1 = 0; soldiers_p1 < max_soldiers_p1 + 1; soldiers_p1++)
        {
            for (int soldiers_p2 = 0; soldiers_p2 < max_soldiers_p2 + 1; soldiers_p2++)
            {
                int num_actions_p1 = battlefield_games[i].num_actions_p1[soldiers_p1];
                int num_actions_p2 = battlefield_games[i].num_actions_p2[soldiers_p2];

                for (int action_p1 = 0; action_p1 < num_actions_p1; action_p1++)
                {
                    for (int action_p2 = 0; action_p2 < num_actions_p2; action_p2++)
                    {
                        int seq_p1 = bf_and_used_soldiers_to_start_seq_id_p1[std::pair<int, int>(i, soldiers_p1)] + action_p1;
                        int seq_p2 = bf_and_used_soldiers_to_start_seq_id_p2[std::pair<int, int>(i, soldiers_p2)] + action_p2;

                        payoff += strat_p1[seq_p1] * strat_p2[seq_p2] * get_payoffs(i, soldiers_p1, soldiers_p2, action_p1, action_p2);
                    }
                }
            }
        }
    }
    return payoff;
}

void BlottoAlt::get_loss_vector_pl1(double *strat_pl2, double *storage)
{
    std::fill(storage, storage + rm_p1->size(), 0.0);
    for (auto &leaf : leaves)
    {
        storage[leaf.seq_pl1()] -= leaf.payoff_pl1() * strat_pl2[leaf.seq_pl2()];
    }
}

void BlottoAlt::get_loss_vector_pl2(double *strat_pl1, double *storage)
{
    std::fill(storage, storage + rm_p2->size(), 0.0);
    for (auto &leaf : leaves)
    {
        storage[leaf.seq_pl2()] -= leaf.payoff_pl2() * strat_pl1[leaf.seq_pl1()];
    }
}

void BlottoAlt::solve(unsigned int num_iterations, bool verbose)
{
    double SADDLE_POINT_GAP_TERMINATE = 0.002;
    // TODO: fix memory leak.
    int num_seqs_p1 = rm_p1->size();
    int num_seqs_p2 = rm_p2->size();

    auto sketchpad_p1 = new double[num_seqs_p1];
    auto sketchpad_p2 = new double[num_seqs_p2];

    auto loss_vector_p1 = new double[num_seqs_p1];
    auto loss_vector_p2 = new double[num_seqs_p2];

    auto accum_strategy_p1 = new double[num_seqs_p1];
    auto accum_strategy_p2 = new double[num_seqs_p2];

    double weight_p1 = 0.0;
    double weight_p2 = 0.0;

    std::fill(accum_strategy_p1, accum_strategy_p1 + num_seqs_p1, 0.0);
    std::fill(accum_strategy_p2, accum_strategy_p2 + num_seqs_p2, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    for (i = 0; i < num_iterations; i++)
    {
        // Bookkeeping every n iterations to decide whether
        // to terminate, and to log saddle point gaps.
        if (i % 100 == 0 && i > 0)
        {
            std::cout << "Iteration: " << i << "\n";
            auto cur = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = cur - start;
            std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

            // Compute loss
            get_loss_vector_pl1(accum_strategy_p2, loss_vector_p1);
            get_loss_vector_pl2(accum_strategy_p1, loss_vector_p2);

            // Compute best response to accumulated strategy
            double br_p1_to_p2 = -rm_p1->best_response(loss_vector_p1, sketchpad_p1);
            double br_p2_to_p1 = rm_p2->best_response(loss_vector_p2, sketchpad_p2);

            br_p1_to_p2 /= weight_p1;
            br_p2_to_p1 /= weight_p2;

            assertm(br_p1_to_p2 > br_p2_to_p1, "Best response to player 1's strategy should be greater than player 2's strategy.");
            double gap = br_p1_to_p2 - br_p2_to_p1;
            std::cout
                << "gap: " << gap << " " << br_p1_to_p2 << " " << br_p2_to_p1 << "\n";

            if (gap < SADDLE_POINT_GAP_TERMINATE)
            {
                std::cout << "Converged!\n";
                break;
            }
        }

        // Do one round of alternation.
        rm_p1->recommend(sketchpad_p1, true);
        for (int j = 0; j < num_seqs_p1; j++)
            accum_strategy_p1[j] += (i + 1) * (i + 1) * sketchpad_p1[j];
        weight_p1 += (i + 1) * (i + 1);
        std::fill(loss_vector_p2, loss_vector_p2 + num_seqs_p2, 0.0);
        for (auto &leaf : leaves)
        {
            loss_vector_p2[leaf.seq_pl2()] -= leaf.payoff_pl2() * sketchpad_p1[leaf.seq_pl1()];
        }
        rm_p2->observe_loss(loss_vector_p2, sketchpad_p2);

        rm_p2->recommend(sketchpad_p2, true);
        for (int j = 0; j < num_seqs_p2; j++)
            accum_strategy_p2[j] += (i + 1) * (i + 1) * sketchpad_p2[j];
        weight_p2 += (i + 1) * (i + 1);
        std::fill(loss_vector_p1, loss_vector_p1 + num_seqs_p1, 0.0);
        for (auto &leaf : leaves)
        {
            loss_vector_p1[leaf.seq_pl1()] -= leaf.payoff_pl1() * sketchpad_p2[leaf.seq_pl2()];
        }
        rm_p1->observe_loss(loss_vector_p1, sketchpad_p1);
    }

    // Extract average straetgy
    for (int j = 0; j < num_seqs_p1; j++)
        accum_strategy_p1[j] /= weight_p1;
    for (int j = 0; j < num_seqs_p2; j++)
        accum_strategy_p2[j] /= weight_p2;

    ret_strat_p1 = accum_strategy_p1;
    ret_strat_p2 = accum_strategy_p2;
}

void BlottoAlt::construct_xi_polytope(int player_id, // 0 or 1
                                      RegretMinimizerEfceFlattenedBase *rm,
                                      std::map<std::pair<int, int>, int> *infosets_start_seq,
                                      std::map<std::pair<int, int>, int> *dummy_start_seq)
{

    int num_soldiers = (player_id == 0) ? max_soldiers_p1 : max_soldiers_p2;

    // Add in basic infosets
    for (int i = 0; i < num_battlefields; i++)
    {
        for (int j = 0; j <= num_soldiers; j++)
        {
            if (i == 0 && j < num_soldiers)
                continue;

            // Compute parent infosets based on whether this is the first battlefield.
            if (i == 0)
            {
                (*infosets_start_seq)[std::pair<int, int>(i, j)] = rm->size();
                rm->add_simplex(0, rm->size(), num_soldiers + 1);
            }
            else
            {
                std::vector<unsigned int> parent_seqs;
                // Iterate over all possible previous infosets to get parents
                for (int num_initial_soldiers = j; num_initial_soldiers <= num_soldiers; num_initial_soldiers++)
                {
                    if (i == 1 && num_initial_soldiers != num_soldiers)
                        continue;
                    int num_soldiers_used = num_initial_soldiers - j;
                    parent_seqs.push_back((*infosets_start_seq)[std::pair<int, int>(i - 1, num_initial_soldiers)] + num_soldiers_used);
                }

                (*infosets_start_seq)[std::pair<int, int>(i, j)] = rm->size();
                rm->add_simplex(parent_seqs, rm->size(), j + 1);
            }
        }
    }

    // Add in dummy infosets
    for (int i = 0; i < num_battlefields; i++)
    {
        for (int num_soldiers_used = 0; num_soldiers_used <= num_soldiers; num_soldiers_used++)
        {
            std::vector<unsigned int> parent_seqs;

            // If battlefield 1, only one infoset with one parent
            if (i == 0)
            {
                parent_seqs.push_back((*infosets_start_seq)[std::pair<int, int>(i, num_soldiers)] + num_soldiers_used);
            }
            else
            {
                // Iterate over all possible previous infosets to get parents
                for (int num_initial_soldiers = num_soldiers_used; num_initial_soldiers <= num_soldiers; num_initial_soldiers++)
                {
                    parent_seqs.push_back((*infosets_start_seq)[std::pair<int, int>(i, num_initial_soldiers)] + num_soldiers_used);
                }
            }

            int num_actions = -1;
            if (player_id == 0)
            {
                num_actions = battlefield_games[i].num_actions_p1[num_soldiers_used];
            }
            else
            {
                num_actions = battlefield_games[i].num_actions_p2[num_soldiers_used];
            }

            (*dummy_start_seq)[std::pair<int, int>(i, num_soldiers_used)] = rm->size();
            rm->add_simplex(parent_seqs, rm->size(), num_actions);
        }
    }
    rm->allocate_mem();
}

void BlottoAlt::construct_leaves()
{
    for (int i = 0; i < num_battlefields; i++)
    {
        for (int j = 0; j <= max_soldiers_p1; j++)
        {
            for (int k = 0; k <= max_soldiers_p2; k++)
            {
                int start_seq_id_p1 = bf_and_used_soldiers_to_start_seq_id_p1[std::pair<int, int>(i, j)];
                int start_seq_id_p2 = bf_and_used_soldiers_to_start_seq_id_p2[std::pair<int, int>(i, k)];
                for (int action_p1 = 0; action_p1 < battlefield_games[i].num_actions_p1[j]; action_p1++)
                {
                    for (int action_p2 = 0; action_p2 < battlefield_games[i].num_actions_p2[k]; action_p2++)
                    {
                        double payoff = battlefield_games[i].get_payoffs(j, k, action_p1, action_p2);
                        int seq_p1 = start_seq_id_p1 + action_p1;
                        int seq_p2 = start_seq_id_p2 + action_p2;
                        leaves.push_back(Leaf(seq_p1, seq_p2, payoff, -payoff, 1.0));
                    }
                }
            }
        }
    }
}