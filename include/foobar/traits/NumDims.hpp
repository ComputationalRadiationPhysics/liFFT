#pragma once

namespace foobar {
namespace traits {

    /**
     * Returns the number of dimensions for the given array-like type
     */
    template< typename T >
    struct NumDims: std::integral_constant< unsigned, T::numDims >{};

}  // namespace traits
}  // namespace foobar
