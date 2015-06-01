#pragma once

namespace foobar {
namespace policies {

    /**
     * Policy that provides functions to get the real and (when applicable) imaginary part
     * for a given index in the given memory structure
     */
    template< typename T_Memory >
    struct GetValue;

}  // namespace policies
}  // namespace foobar
