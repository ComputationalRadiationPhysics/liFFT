#pragma once

namespace foobar {
namespace traits {

    /**
     * Returns the memory type of a given data structure
     */
    template< typename T_Data >
    struct MemoryType
    {
        using type = typename T_Data::Memory;
    };

}  // namespace traits
}  // namespace foobar
