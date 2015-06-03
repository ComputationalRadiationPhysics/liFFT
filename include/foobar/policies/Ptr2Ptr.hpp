#pragma once

#include <boost/utility.hpp>

namespace foobar {
namespace policies {

    /**
     * Converts one pointer type into another
     */
    template< typename T_Src, typename T_Dest = T_Src >
    struct Ptr2Ptr
    {
        using Src = T_Src;
        using Dest = T_Dest;

        Dest*
        operator()(Src* data)
        {
            return reinterpret_cast<Dest*>(data);
        }

        const Dest*
        operator()(const Src* data)
        {
            return reinterpret_cast<const Dest*>(data);
        }
    };

}  // namespace policies
}  // namespace foobar
