#pragma once

#include <cassert>

namespace foobar{
namespace types{

    template< typename T, unsigned T_numDims >
    class Vec
    {
    public:
        static constexpr unsigned numDims = T_numDims;
        using type = T;

        template<
           typename... TArgs,
           typename = std::enable_if_t<(sizeof...(TArgs) == (numDims-1))>
           >
            Vec(T val, TArgs ... values): values_{std::forward<T>(val), std::forward<TArgs>(values)...}
        {}

        Vec(){}

        T&
        operator[](unsigned dim)
        {
            assert(dim>=0 && dim<numDims);
            return values_[dim];
        }

        T
        operator[](unsigned dim) const
        {
            assert(dim>=0 && dim<numDims);
            return values_[dim];
        }

        T*
        data(){
            return values_;
        }

    private:
        T values_[numDims];
    };

    template< typename T, T T_val, unsigned T_numDims >
    struct ConstVec
    {
        static constexpr unsigned numDims = T_numDims;
        using type = T;

        constexpr T
        operator[](unsigned dim)
        {
            return (dim>=0 && dim<numDims) ? T_val : throw std::logic_error("Out of range");
        }
    };

} // namespace types
} // namespace foobar
