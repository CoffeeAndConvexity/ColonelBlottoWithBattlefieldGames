#pragma once

#include <vector>
#include "id_types.h"
#include "util.h"

#include "regret_minimizer.h"

class RegretMinimizerEfceFlattenedBase : public RegretMinimizer
{
protected:
    struct Act
    {
        unsigned int simplex_index;
        unsigned int simplex_size;
        unsigned int h_start;
        unsigned int h_end;
    };

    struct Constant
    {
        unsigned int index;
        double value;
    };

    // Top-down decomposition of the simplices which are added.
    std::vector<Act> operations_;

    // Constants, if any
    std::vector<Constant> constants_;

    // Vector of h_indices. Note that coefficients are not required
    // since these are all equal to 1.
    std::vector<unsigned int> h_indices_;

    // Size of the *entire* regret minimizer. May be set at the beginning
    // if size is known, if not, will be updated by calls to add_simplex().
    unsigned int size_;

    // Flag to indicate if we have allocated memory (or more generally, initialized)
    bool has_allocated_regret_mem_;

    // Test if regret_ has been allocated.
    bool has_allocated_regret_mem() const;

public:
    // Creates regret minimizer over a polytope with known size.
    RegretMinimizerEfceFlattenedBase(unsigned int size);

    RegretMinimizerEfceFlattenedBase();

    ~RegretMinimizerEfceFlattenedBase();
    virtual void recommend(double *store, bool unused) const = 0;
    virtual void observe_loss(double *loss, double *old_result) = 0;
    double best_response(double *loss, double *old_result);

    void add_simplex(std::vector<unsigned int> &h_indices,
                     unsigned int simplex_index,
                     unsigned int simplex_size);
    void add_simplex(unsigned int singleton_h_index,
                     unsigned int simplex_index,
                     unsigned int simplex_size);
    void add_constant(unsigned int index, double value);

    // Allocates memory for `regret_`. Call this when we have completed
    // constructing the regret mininimizer and are guaranteed to not
    // have to call add_simplex() again.
    virtual void allocate_mem() = 0;

    void log_memory();

    // Gets size of the entire regret minimizer.
    const unsigned int size() const;
};
