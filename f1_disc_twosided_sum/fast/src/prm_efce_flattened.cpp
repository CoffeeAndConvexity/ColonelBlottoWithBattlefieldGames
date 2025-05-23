#include "prm_efce_flattened.h"
#include <iostream>
#include "logger.h"
#include <cstdlib>

PrmEfceFlattened::PrmEfceFlattened(unsigned int size)
    : RegretMinimizerEfceFlattenedBase(size)
{
}

PrmEfceFlattened::PrmEfceFlattened()
    : RegretMinimizerEfceFlattenedBase(1), regret_(NULL)
{
}

PrmEfceFlattened::~PrmEfceFlattened()
{
    delete[] regret_;
    delete[] scratchpad;
}

void PrmEfceFlattened::recommend(double *store, bool unused) const
{
    // Compute strategy to recommend if this was one stage.
    store[0] = 1.0;

    for (const auto &cst : constants_)
    {
        store[cst.index] = cst.value;
    }

    for (const auto &act : operations_)
    {
        // Use rm on simplex to get the first step //
        double t = 0.0;
        for (unsigned int idx = act.simplex_index;
             idx < act.simplex_index + act.simplex_size;
             idx++)
        {
            if (regret_[idx] < 0)
            {
                continue;
            }
            t += regret_[idx];
        }

        for (unsigned int idx = act.simplex_index;
             idx < act.simplex_index + act.simplex_size;
             idx++)
        {
            if (t > 0)
            {
                store[idx] = std::max(0.0, regret_[idx] / t);
            }
            else
            {
                store[idx] = 1.0 / (double)act.simplex_size;
            }
        }

        // =====================================================
        // Pretend to get another round of losses and use store
        // to temporarily store the fake regrets
        double fake_loss = inner_product(&scratchpad[act.simplex_index],
                                         &store[act.simplex_index],
                                         act.simplex_size, 0.0);
        for (unsigned int idx = act.simplex_index;
             idx < act.simplex_index + act.simplex_size;
             idx++)
        {
            store[idx] = regret_[idx] + fake_loss - scratchpad[idx];
        }

        t = 0.0;
        // Run another round of RM on these new regrets
        for (unsigned int idx = act.simplex_index;
             idx < act.simplex_index + act.simplex_size;
             idx++)
        {
            if (store[idx] < 0)
            {
                continue;
            }
            t += store[idx];
        }
        for (unsigned int idx = act.simplex_index;
             idx < act.simplex_index + act.simplex_size;
             idx++)
        {
            if (t > 0)
            {
                store[idx] = std::max(0.0, store[idx] / t);
            }
            else
            {
                store[idx] = 1.0 / (double)act.simplex_size;
            }
        }

        // store PROBABILITY DISTRIBUTION into scratchpad
        for (unsigned int idx = act.simplex_index;
             idx < act.simplex_index + act.simplex_size;
             idx++)
        {
            scratchpad[idx] = store[idx];
        }
        // ===================================================

        // Do scaled operator
        double scale = 0.0;
        for (unsigned int h_idx = act.h_start;
             h_idx < act.h_end;
             h_idx++)
        {
            scale += store[h_indices_[h_idx]];
        }

        for (unsigned int idx = act.simplex_index;
             idx < act.simplex_index + act.simplex_size;
             idx++)
        {
            store[idx] *= scale;
        }
    }
}

void PrmEfceFlattened::observe_loss(double *loss, double *old_result)
{
    // Warning:  Modifies old result.
    for (std::vector<Act>::reverse_iterator i = operations_.rbegin();
         i != operations_.rend();
         ++i)
    {
        old_result = scratchpad;
        // observe loss on simplex
        double loss_obtained = inner_product(&loss[i->simplex_index],
                                             &old_result[i->simplex_index],
                                             i->simplex_size,
                                             0.0);

        for (unsigned int idx = i->simplex_index;
             idx < i->simplex_index + i->simplex_size;
             idx++)
        {
            regret_[idx] += loss_obtained - loss[idx];
        }

        // adjust loss for parents based on h index
        for (unsigned int h_idx = i->h_start;
             h_idx < i->h_end;
             h_idx++)
        {
            loss[h_indices_[h_idx]] += loss_obtained;
        }
    }

    // Copy loss into scratchpad, needed to recommend actions later on.
    // Note that this loss already accounts for children.
    for (unsigned int idx = 0; idx < size_; ++idx)
    {
        scratchpad[idx] = loss[idx];
    }
}

void PrmEfceFlattened::allocate_mem()
{
    assertm(has_allocated_regret_mem() == false, "Memory has already been allocated!");
    assertm(regret_ == NULL, "Memory has already been allocated!");
    regret_ = new double[size_];
    scratchpad = new double[size_];
    std::fill(regret_, regret_ + size_, 0.0);
    std::fill(scratchpad, scratchpad + size_, 0.0);
}
