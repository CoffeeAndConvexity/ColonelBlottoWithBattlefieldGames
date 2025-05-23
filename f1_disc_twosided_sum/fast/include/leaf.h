#pragma once

#include "util.h"

class Leaf
{
    const SeqId _seq_pl1;
    const SeqId _seq_pl2;
    const double _payoff_pl1;
    const double _payoff_pl2;
    const double _chance_factor;

public:
    Leaf(SeqId seq_pl1, SeqId seq_pl2,
         double payoff_pl1, double payoff_pl2,
         double chance_factor);

    const SeqId seq_pl1() const;
    const SeqId seq_pl2() const;
    const SeqId seq_id(Player player_id) const;
    const SeqPair seq_pair() const;

    const double payoff_pl1() const;
    const double payoff_pl2() const;
    const double payoff(Player player_id) const;

    const double chance_factor() const;
};