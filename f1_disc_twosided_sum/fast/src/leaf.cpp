#include "leaf.h"

Leaf::Leaf(SeqId seq_pl1, SeqId seq_pl2,
           double payoff_pl1, double payoff_pl2,
           double chance_factor) : _seq_pl1(seq_pl1),
                                   _seq_pl2(seq_pl2),
                                   _payoff_pl1(payoff_pl1),
                                   _payoff_pl2(payoff_pl2),
                                   _chance_factor(chance_factor)
{
    assertm(chance_factor >= 0.0, "Leaf chance factor required to be non-negative");
    assertm(chance_factor <= 1.0, "Leaf chance factor required to be less than 1.0");
}

const SeqId Leaf::seq_pl1() const
{
    return _seq_pl1;
}
const SeqId Leaf::seq_pl2() const
{
    return _seq_pl2;
}

const SeqId Leaf::seq_id(Player player_id) const
{
    if (player_id == Player::pl1)
    {
        return _seq_pl1;
    }
    else if (player_id == Player::pl2)
    {
        return _seq_pl2;
    }
    else
    {
        assertm(false, "Invalid player id");
    }
}

const SeqPair Leaf::seq_pair() const
{
    return SeqPair(_seq_pl1, _seq_pl2);
}

const double Leaf::payoff_pl1() const
{
    return _payoff_pl1;
}

const double Leaf::payoff_pl2() const
{
    return _payoff_pl2;
}

const double Leaf::payoff(Player player_id) const
{
    if (player_id == Player::pl1)
    {
        return _payoff_pl1;
    }
    else if (player_id == Player::pl2)
    {
        return _payoff_pl2;
    }
    else
    {
        assertm(false, "Invalid player id");
    }
}

const double Leaf::chance_factor() const
{
    return _chance_factor;
}