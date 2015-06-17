#pragma once

#include <boost/utility.hpp>

namespace foobar {
namespace policies {

    /**
     * Functor to update the external data of a container (after the internal data was modified) if required (defaults to No-Op)
     */
    template< typename T_Data >
    struct WriteData
    {
        using Data = T_Data;

        void operator()(Data& data){}
    };

}  // namespace policies
}  // namespace foobar
