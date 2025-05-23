// Extracted from
// https://stackoverflow.com/questions/30469063/extract-a-subvector-from-a-vector-without-copy
// to return subarrays.

#pragma once

template <typename Iterator>
class slice_view
{
    Iterator begin_;
    Iterator end_;

public:
    slice_view(Iterator begin, Iterator end) : begin_(begin), end_(end) {}
    Iterator begin() const { return this->begin_; }
    Iterator end() const { return this->end_; }
};

template <typename Iterator>
class array_view : public slice_view<Iterator>
{
public:
    array_view(Iterator begin, Iterator end) : slice_view<Iterator>(begin, end) {}
    typename std::iterator_traits<Iterator>::reference
    operator[](std::size_t index) { return this->begin_[index]; }
};

template <typename Iterator>
class enum_view
{
private:
    struct enum_iter
    {
        size_t i;
        Iterator iter;

        bool operator!=(const enum_iter &other) const { return iter != other.iter; }
        void operator++()
        {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    size_t enum_start_;
    Iterator begin_;
    Iterator end_;

public:
    auto begin() { return enum_iter{enum_start_, begin_}; }
    auto end() { return enum_iter{enum_start_, end_}; }
    enum_view(size_t enum_start, Iterator begin, Iterator end) : enum_start_(enum_start), begin_(begin), end_(end) {}
};

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type>
constexpr auto enumerate_slice(size_t enum_start, Iterator begin, Iterator end)
{
    return enum_view<Iterator>(enum_start, begin, end);
}
