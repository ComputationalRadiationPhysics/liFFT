#pragma once

namespace foobar {

    /** Commonly used pattern to silence unused variable warnings */
    template <typename... T>
    void ignore_unused(const T& ...){}

}  // namespace foobar
