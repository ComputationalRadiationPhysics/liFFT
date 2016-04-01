/* This file is part of HaLT.
 *
 * HaLT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * HaLT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with HaLT.  If not, see <www.gnu.org/licenses/>.
 */
 
#pragma once

#include <type_traits>
#include "haLT/traits/RemoveModifiers.hpp"
#include "haLT/void_t.hpp"
#include "haLT/c++14_types.hpp"

namespace haLT {
namespace traits {

    // Forward declaration
    template< typename T >
    struct IntegralType;

    /**
     * Specialize this to return the integral type of a given memory representation
     */
    template< typename T, typename T_SFINAE = void >
    struct IntegralTypeImpl;

    template< typename T >
    struct IntegralTypeImpl<
            T,
            std::enable_if_t<
                (std::is_integral<T>::value || std::is_floating_point<T>::value)
            >
        >
    {
        using type = T;
    };

    /**
     * Specialization for structs with a type-member
     * Hint: More Specializations on differently named type aliases are possible with the void_t
     */
    template< typename T >
    struct IntegralTypeImpl< T, void_t< typename T::type > >: IntegralType< typename T::type >{};

    /**
     * Returns the integral type of a given memory representation
     */
    template< typename T >
    struct IntegralType{
        // Removes all modifiers and calls IntegralTypeImpl
        using type = typename IntegralTypeImpl<
                typename RemoveModifer<T>::type
              >::type;
    };

    template< typename T >
    using IntegralType_t = typename IntegralType<T>::type;

}  // namespace traits
}  // namespace haLT
