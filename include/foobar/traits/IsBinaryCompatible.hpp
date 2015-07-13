#pragma once

#include "foobar/c++14_types.hpp"

namespace foobar {
namespace traits {

    /**
     * Implementation of \ref IsBinaryCompatible
     * Specialize this!
     */
    template< typename T_Src, typename T_Dest >
    struct IsBinaryCompatibleImpl: std::false_type{};

    template< typename T >
    struct IsBinaryCompatibleImpl< T, T >: std::true_type{};

    /**
     * Returns true if this types are binary compatible,
     * that is a conversion between pointers to those types is "safe"
     */
    template< typename T_Src, typename T_Dest >
    struct IsBinaryCompatible
            : std::integral_constant<
              bool,
              IsBinaryCompatibleImpl< std::remove_cv_t<T_Src>, std::remove_cv_t<T_Dest> >::value ||
              IsBinaryCompatibleImpl< std::remove_cv_t<T_Dest>, std::remove_cv_t<T_Src> >::value
              >{};

}  // namespace traits
}  // namespace foobar
