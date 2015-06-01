#pragma once

namespace foobar {
namespace policies {

    /**
     * A functor that is applied to a reference to T_Memory and returns the intensity at a given index
     */
    template< class T_Memory >
    struct GetIntensity;

}  // namespace policies
}  // namespace foobar
