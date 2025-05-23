#include "rm_efce_flattened.h"
#include <iostream>
#include "logger.h"
#include <cstdlib>

RmEfceFlattened::RmEfceFlattened(unsigned int size)
    : RegretMinimizerEfceFlattenedBase(size)
{
}

RmEfceFlattened::RmEfceFlattened()
    : RegretMinimizerEfceFlattenedBase(1), regret_(NULL)
{
}

RmEfceFlattened::~RmEfceFlattened()
{
    delete[] regret_;
}

void RmEfceFlattened::recommend(double *store, bool unused) const
{
    store[0] = 1.0;

    for (const auto &cst : constants_)
    {
        store[cst.index] = cst.value;
    }

    for (const auto &act : operations_)
    {
        // Use rm on simplex //
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

void RmEfceFlattened::observe_loss(double *loss, double *old_result)
{
    // Warning:  Modifies old result.
    for (std::vector<Act>::reverse_iterator i = operations_.rbegin();
         i != operations_.rend();
         ++i)
    {
        // Use rm+ on simplex ---- recompute recommended result (TODO: fix).//
        double t = 0.0;
        for (unsigned int idx = i->simplex_index;
             idx < i->simplex_index + i->simplex_size;
             idx++)
        {
            if (regret_[idx] < 0)
            {
                continue;
            }
            t += regret_[idx];
        }

        for (unsigned int idx = i->simplex_index;
             idx < i->simplex_index + i->simplex_size;
             idx++)
        {
            if (t > 0)
            {
                old_result[idx] = std::max(0.0, regret_[idx] / t);
            }
            else
            {
                old_result[idx] = 1.0 / (double)i->simplex_size;
            }
        }

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
}

void RmEfceFlattened::allocate_mem()
{
    assertm(has_allocated_regret_mem() == false, "Memory has already been allocated!");
    assertm(regret_ == NULL, "Memory has already been allocated!");
    regret_ = new double[size_];
    std::fill(regret_, regret_ + size_, 0.0);
}
