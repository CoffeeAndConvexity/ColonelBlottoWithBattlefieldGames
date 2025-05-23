#pragma once

#include "range.hpp"
#include "enumerate.hpp"
#include <assert.h>
#include <tuple>
#include "id_types.h"

const double NEG_INFINITY = std::numeric_limits<double>::min();

using namespace util::lang;
// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))
template <typename T,
          typename H = decltype(std::declval<T>() * std::declval<T>())>
T inner_product(T *a, T *b, unsigned int size, H init_val)
{
    H val = init_val;
    for (auto i : range(0, size))
    {
        val += a[i] * b[i];
    }
    return val;
}

template <typename T>
void vector_sum(const T *a, const T *b, T *dst, unsigned int size)
{
    for (auto i : range(0, size))
    {
        dst[i] = a[i] + b[i];
    }
}

template <typename T>
void vector_sum(const T *a, const T b, T *dst, unsigned int size)
{
    for (auto i : range(0, size))
    {
        dst[i] = a[i] + b;
    }
}

template <typename T>
void vector_subtract(const T *a, const T *b, T *dst, unsigned int size)
{
    for (auto i : range(0, size))
    {
        dst[i] = a[i] - b[i];
    }
}

inline Player operator-(Player a)
{
    if (a == Player::pl1)
    {
        return Player::pl2;
    }
    else
    {
        return Player::pl1;
    }
}

// Create new sequence pair, but with "rotation" of player ids.
inline SeqPair SeqPairR(SeqId id_1, SeqId id_2, Player first_player)
{
    if (first_player == Player::pl1)
    {
        return SeqPair{id_1, id_2};
    }
    else
    {
        return SeqPair{id_2, id_1};
    }
}

// Create new infoset pair, but with "rotation" of player ids.
inline InfPair InfPairR(InfId id_1, InfId id_2, Player first_player)
{
    if (first_player == Player::pl1)
    {
        return InfPair(id_1, id_2);
    }
    else
    {
        return InfPair(id_2, id_1);
    }
}

// Takes an old sequence pair and returns a new sequence
// pair with the sequence pair for `player_id` replaced
// by some other sequence.
inline SeqPair replace_sequence_pair(SeqPair old_seq_pair, Player player_id, SeqId new_seq_id)
{
    auto [seq_pl1, seq_pl2] = old_seq_pair;
    if (player_id == Player::pl1)
    {
        return SeqPair(new_seq_id, seq_pl2);
    }
    else if (player_id == Player::pl2)
    {
        return SeqPair(seq_pl1, new_seq_id);
    }
    else
    {
        assertm(false, "Invalid player id");
    }
}

// Code for hasing pairs of generics.
namespace std
{
    namespace
    {

        // Code from boost
        // Reciprocal of the golden ratio helps spread entropy
        //     and handles duplicates.
        // See Mike Seymour in magic-numbers-in-boosthash-combine:
        //     https://stackoverflow.com/questions/4948780

        template <class T>
        inline void hash_combine(std::size_t &seed, T const &v)
        {
            seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl
        {
            static void apply(size_t &seed, Tuple const &tuple)
            {
                HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
                hash_combine(seed, get<Index>(tuple));
            }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple, 0>
        {
            static void apply(size_t &seed, Tuple const &tuple)
            {
                hash_combine(seed, get<0>(tuple));
            }
        };
    }

    template <typename... TT>
    struct hash<std::tuple<TT...>>
    {
        size_t
        operator()(std::tuple<TT...> const &tt) const
        {
            size_t seed = 0;
            HashValueImpl<std::tuple<TT...>>::apply(seed, tt);
            return seed;
        }
    };
}