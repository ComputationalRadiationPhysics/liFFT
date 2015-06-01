#pragma once

namespace foobar {
namespace traits {

    /**
     * Removes all modifiers (const, volatile, reference, pointer, array extents) from the given type
     */
    template< typename T >
    struct RemoveModifer{
        using type = typename std::remove_pointer<
                         typename std::remove_reference<
                             typename std::remove_all_extents<
                                 typename std::remove_cv<T>::type
                             >::type
                         >::type
                     >::type;
    };
}  // namespace traits
}  // namespace foobar
