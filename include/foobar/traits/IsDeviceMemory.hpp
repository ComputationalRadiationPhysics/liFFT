#pragma once

#include <cufft.h>

namespace foobar {
namespace libraries {
namespace traits{

        /**
         * Returns whether the given data container is already on the device or not
         * A true_type implies that GetRawPtr returns a device pointer
         */
        template< typename T >
        struct IsDeviceMemory: std::false_type{};

}  // namespace traits
}  // namespace libraries
}  // namespace foobar
