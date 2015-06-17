#pragma once

namespace foobar {
namespace types {

    template< typename T1, typename T2 >
    struct TypePair
    {
        using First = T1;
        using Second = T2;
    };

}  // namespace types
}  // namespace foobar
