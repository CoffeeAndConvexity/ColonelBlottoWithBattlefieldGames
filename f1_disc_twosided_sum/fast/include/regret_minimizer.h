#pragma once

#include "util.h"
#include "id_types.h"

class RegretMinimizer
{
public:
    virtual void recommend(double *store, bool memorize) const = 0;
    virtual void observe_loss(double *loss, double *sketch_start) = 0;
    virtual const unsigned int size() const = 0;
    virtual ~RegretMinimizer(){};
};