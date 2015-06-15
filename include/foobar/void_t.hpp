#pragma once

namespace foobar {

    /**
     * Helper for void_t (Workaround, see Paper from Walter Brown)
     */
    template <typename...>
    struct voider { using type = void; };

    /**
     * void_t for evaluating arguments, then returning void
     * Used for SFINAE evaluation of types
     */
    template <typename... Ts>
    using void_t = typename voider<Ts...>::type;

}  // namespace foobar
