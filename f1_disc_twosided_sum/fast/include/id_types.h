#pragma once

#include <limits>

enum Player
{
    pl1 = 0,
    pl2 = 1,
};

typedef unsigned int SeqId;
typedef unsigned int InfId;
typedef unsigned int SubgameId;

const SeqId EMPTY_SEQUENCE_ID = 0;
const InfId MAX_INF_ID = std::numeric_limits<unsigned int>::max();

typedef std::tuple<SeqId, SeqId> SeqPair;
typedef std::tuple<InfId, InfId> InfPair;
