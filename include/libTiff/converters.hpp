#pragma once

#include <type_traits>
#include <limits>
#include "foobar/c++14_types.hpp"

namespace libTiff {

    /**
     * Converts a single channel
     * FP -> FP: values are unchanged
     */
    template<
        typename T_Src,
        typename T_Dest,
        bool T_srcIsFloat = std::is_floating_point<T_Src>::value,
        bool T_destIsFloat = std::is_floating_point<T_Dest>::value
    >
    struct ConvertChannel
    {
        static_assert(std::is_floating_point<T_Src>::value, "Invalid src type");
        static_assert(std::is_floating_point<T_Dest>::value, "Invalid dst type");

        using Src = T_Src;

        T_Dest
        operator()(T_Src src)
        {
            return src;
        }
    };

    /**
     * Integral -> FP: normalize to [0, 1]
     */
    template< typename T_Src, typename T_Dest >
    struct ConvertChannel< T_Src, T_Dest, false, true >
    {
        static_assert(std::is_integral<T_Src>::value, "Invalid src type");
        static_assert(std::is_floating_point<T_Dest>::value, "Invalid dst type");

        using Src = T_Src;

        static constexpr T_Dest divisor = T_Dest(std::numeric_limits<T_Src>::max()) - std::numeric_limits<T_Src>::min();
        static constexpr T_Dest factor = 1/divisor;

        static constexpr T_Dest min = std::numeric_limits<T_Src>::min();

        T_Dest
        operator()(T_Src src)
        {
            return (src - min) * factor;
        }
    };

    /**
     * Integral -> Integral: Extent from one range to another
     */
    template< typename T_Src, typename T_Dest >
    struct ConvertChannel< T_Src, T_Dest, false, false >
    {
        using Src = T_Src;
        using Dest = T_Dest;

        static_assert(std::is_integral<Src>::value, "Invalid src type");
        static_assert(std::is_integral<Dest>::value, "Invalid dst type");

        using Divisior = std::make_unsigned_t<Src>;
        static constexpr Divisior divisor = Divisior(std::numeric_limits<Src>::max()) - std::numeric_limits<Src>::min();

        static constexpr Dest max = std::numeric_limits<Dest>::max();
        static constexpr Dest min = std::numeric_limits<Dest>::min();

        Dest
        operator()(Src src)
        {
            return static_cast<Dest>(src * max / divisor - min);
        }
    };

    /**
     * FP -> Integral: Assume normalized FP [0,1] and scale accordingly
     */
    template< typename T_Src, typename T_Dest >
    struct ConvertChannel< T_Src, T_Dest, true, false >
    {
        using Src = T_Src;
        using Dest = T_Dest;

        static_assert(std::is_floating_point<Src>::value, "Invalid src type");
        static_assert(std::is_integral<Dest>::value, "Invalid dst type");

        static constexpr Dest max = std::numeric_limits<Dest>::max();
        static constexpr Dest min = std::numeric_limits<Dest>::min();

        Dest
        operator()(Src src)
        {
            return static_cast<Dest>(src * max - min);
        }
    };

    template< typename T_Src, typename T_Dest >
    struct ConvertARGBToMono
    {
        using Src = T_Src;

        using FloatType =
                std::conditional_t<
                    std::is_same<T_Src, double>::value || std::is_same<T_Dest, double>::value,
                    double,
                    float
                >;

        static constexpr FloatType rWeight = 0.299;
        static constexpr FloatType gWeight = 0.587;
        static constexpr FloatType bWeight = 0.114;

        T_Dest
        operator()(T_Src a, T_Src r, T_Src g, T_Src b)
        {
            return static_cast<T_Dest>(rWeight * r + gWeight * g + bWeight * b);
        }
    };

    template< typename T_El, class T_Func >
    struct AccessARGB
    {
        T_Func func_;

        template< typename T_Src >
        auto
        operator()(const T_Src& src)
        -> std::result_of_t<T_Func(T_El, T_El, T_El, T_El)>
        {
            const T_El* els = reinterpret_cast<const T_El*>(&src);
            return func_(els[0], els[1], els[2], els[3]);
        }
    };

    template< typename T_Src, class T_Func >
    struct ConvertAllChannels
    {
        using Src = T_Src;

        using Dest = std::result_of_t<T_Func(T_Src)>;
        using Result = std::array<Dest, 4>;
        T_Func func_;

        Result
        operator()(T_Src a, T_Src r, T_Src g, T_Src b)
        {
            Result res;
            res[0] = func_(a);
            res[1] = func_(r);
            res[2] = func_(g);
            res[3] = func_(b);
            return res;
        }
    };

    template< typename T_Src, typename T_Dest, bool T_fromARGB, bool T_toARGB >
    struct Convert;

    template< typename T_Src, typename T_Dest >
    struct Convert< T_Src, T_Dest, false, false >: ConvertChannel<T_Src, T_Dest>{};

    template< typename T_Src, typename T_Dest >
    struct Convert< T_Src, T_Dest, false, true >
    {
        using Src = T_Src;
        using Dest = T_Dest;
        using Result = std::array<Dest, 4>;
        ConvertChannel<Src, Dest> conv_;

        Result
        operator()(Src src)
        {
            Result res;
            res.fill(conv_(src));
            return res;
        }
    };

    template< typename T_Src, typename T_Dest >
    struct Convert< T_Src, T_Dest, true, false >:
        AccessARGB<
            T_Src,
            ConvertARGBToMono< T_Src, T_Dest >
        >
    {
        using Src = T_Src;
        using Dest = T_Dest;
    };

    template< typename T_Src, typename T_Dest >
    struct Convert< T_Src, T_Dest, true, true >:
        AccessARGB<
            T_Src,
            ConvertAllChannels<
                T_Src,
                ConvertChannel<T_Src, T_Dest>
            >
        >
    {
        using Src = T_Src;
        using Dest = T_Dest;
    };

}  // namespace libTiff
