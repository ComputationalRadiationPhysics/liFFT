#pragma once

#include <type_traits>

namespace foobar {

    /**
     * Type used to indicate that a given value should be automatically detected
     */
    struct AutoDetect: std::integral_constant<unsigned, 0>{};

}  // namespace foobar
