#pragma once

namespace foobar {
namespace traits{

        /**
         * Returns whether the given data container is already on the device or not
         * A true_type implies that the underlying memory is on the device and therefore a reference is a device pointer
         */
        template< typename T >
        struct IsDeviceMemory: std::false_type{};

}  // namespace traits
}  // namespace foobar
