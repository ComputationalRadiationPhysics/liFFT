#pragma once

namespace foobar {
namespace traits {

    /**
     * Evaluates to true type if the given Array-like type is strided
     */
    template< typename T >
    struct IsStrided: std::integral_constant<bool, T::isStrided>{};

}  // namespace traits
}  // namespace foobar
