#pragma once

namespace foobar {
namespace policies {

    /**
     * Functor to update the internal data of a container if required (defaults to No-Op)
     */
    template< typename T_Data >
    struct ReadData
    {
        using Data = T_Data;

        void operator()(Data& data){}
    };

}  // namespace policies
}  // namespace foobar
