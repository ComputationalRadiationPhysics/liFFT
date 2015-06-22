#pragma once

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
              IsBinaryCompatibleImpl< T_Src, T_Dest >::value || IsBinaryCompatibleImpl< T_Dest, T_Src >::value
              >{};

}  // namespace traits
}  // namespace foobar
