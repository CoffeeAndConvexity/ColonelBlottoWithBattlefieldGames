#pragma once

#include <vector>
#include "id_types.h"
#include "util.h"

#include "regret_minimizer.h"
#include "regret_minimizer_efce_flattened_base.h"

class RmPlusEfceFlattened : public RegretMinimizerEfceFlattenedBase
{
private:
    double *regret_;

public:
    RmPlusEfceFlattened(unsigned int size);

    RmPlusEfceFlattened();

    ~RmPlusEfceFlattened();
    void recommend(double *store, bool unused) const;
    void observe_loss(double *loss, double *old_result);

    // Allocates memory for `regret_`. Call this when we have completed
    // constructing the regret mininimizer and are guaranteed to not
    // have to call add_simplex() again.
    void allocate_mem();
};
