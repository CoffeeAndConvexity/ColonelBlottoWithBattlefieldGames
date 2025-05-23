#include "regret_minimizer_efce_flattened_base.h"
#include <iostream>
#include "logger.h"
#include <cstdlib>

RegretMinimizerEfceFlattenedBase::RegretMinimizerEfceFlattenedBase(unsigned int size)
    : size_(size), has_allocated_regret_mem_(false)
{
}

RegretMinimizerEfceFlattenedBase::RegretMinimizerEfceFlattenedBase()
    : size_(1), has_allocated_regret_mem_(false)
{
}

RegretMinimizerEfceFlattenedBase::~RegretMinimizerEfceFlattenedBase()
{
}

const unsigned int RegretMinimizerEfceFlattenedBase::size() const
{
    return size_;
}

double RegretMinimizerEfceFlattenedBase::best_response(double *loss, double *storage)
{
    std::fill(storage, storage + size_, 0.0);
    // Warning:  Modifies old result.
    for (std::vector<Act>::reverse_iterator i = operations_.rbegin();
         i != operations_.rend();
         ++i)
    {
        double lowest_loss = std::numeric_limits<double>::max();
        int lowest_idx = i->simplex_index;
        for (unsigned int idx = i->simplex_index;
             idx < i->simplex_index + i->simplex_size;
             idx++)
        {
            if (loss[idx] < lowest_loss)
            {
                lowest_loss = loss[idx];
                lowest_idx = idx;
            }
        }

        // adjust loss for parents based on h index
        for (unsigned int h_idx = i->h_start;
             h_idx < i->h_end;
             h_idx++)
        {
            loss[h_indices_[h_idx]] += lowest_loss;
        }
        storage[lowest_idx] = 1.0;
    }

    return loss[0];
}

void RegretMinimizerEfceFlattenedBase::add_simplex(std::vector<unsigned int> &h_indices,
                                                   unsigned int simplex_index, unsigned int simplex_size)
{
    unsigned int start_h_index = h_indices_.size();
    unsigned int end_h_index = start_h_index + h_indices.size();
    for (auto h_index : h_indices)
    {
        h_indices_.push_back(h_index);
    }
    operations_.push_back(Act{simplex_index, simplex_size, start_h_index, end_h_index});

    if (!has_allocated_regret_mem())
        size_ += simplex_size;
}

void RegretMinimizerEfceFlattenedBase::add_simplex(unsigned int singleton_h_index,
                                                   unsigned int simplex_index,
                                                   unsigned int simplex_size)
{
    unsigned int start_h_index = h_indices_.size();
    unsigned int end_h_index = start_h_index + 1;

    h_indices_.push_back(singleton_h_index);
    operations_.push_back(Act{simplex_index, simplex_size, start_h_index, end_h_index});

    if (!has_allocated_regret_mem())
        size_ += simplex_size;
}

void RegretMinimizerEfceFlattenedBase::add_constant(unsigned int index, double value)
{
    constants_.push_back(Constant{index, value});
    if (!has_allocated_regret_mem())
        size_++;
}

bool RegretMinimizerEfceFlattenedBase::has_allocated_regret_mem() const
{
    return has_allocated_regret_mem_;
}

void RegretMinimizerEfceFlattenedBase::log_memory()
{
    LOG(linfo) << "Space for operations: " << operations_.size() * sizeof(Act);
    LOG(linfo) << "Space for indices: " << h_indices_.size() * sizeof(unsigned int);
}
