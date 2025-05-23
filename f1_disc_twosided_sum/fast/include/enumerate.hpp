// Template to enumerate over iterator, python style
// Source adapted from:
// https://www.reedbeta.com/blog/python-like-enumerate-in-cpp17/
// with modifications in terms of starting elements.

#pragma once

#include <tuple>

template <typename T,
          typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T &&iterable, int init_idx = 0)
{
    struct iterator
    {
        size_t i;
        TIter iter;
        bool operator!=(const iterator &other) const { return iter != other.iter; }
        void operator++()
        {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper
    {
        T iterable;
        auto begin() { return iterator{init_idx, std::begin(iterable)}; }
        auto end() { return iterator{init_idx, std::end(iterable)}; }
    };
    return iterable_wrapper{std::forward<T>(iterable)};
}
