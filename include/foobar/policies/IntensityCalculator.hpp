#pragma once

#include "foobar/traits/all.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "IntensityLibrary.hpp"
#include <boost/array.hpp>

namespace foobar {
namespace policies {

    template< class T_Input >
    class IntensityCalculator
    {
    public:
        using Input = T_Input;
        using Output = typename traits::IntegralType<Input>::type*;
        using RawPtr = GetRawPtr<Input>;
        using Extents = GetExtents<Input>;
        using ExtentsPtr = GetExtentsRawPtr<Input>;
        static constexpr bool isComplex = traits::IsComplex<Input>::value;
        static constexpr unsigned numDims = traits::NumDims<Input>::value;
        static_assert(numDims >= 1, "Need >= 1 dimension");
        static constexpr bool isAoS = traits::IsAoS<Input>::value;

    private:
        template< unsigned T_numDims, bool T_isComplex, bool T_isAoS, class DUMMY = void >
        struct ExecutionPolicy;

        // Real 1D
        template< bool T_isAoS >
        struct ExecutionPolicy< 1, false, T_isAoS >
        {
            void operator()(Input& input, Output output){
                LibFoo::calculateR1D(RawPtr()(input), output, Extents(input)[0]);
            }
        };

        // Real ND
        template< unsigned T_numDims, bool T_isAoS >
        struct ExecutionPolicy< T_numDims, false, T_isAoS >
        {
            void operator()(Input& input, Output output){
                ExtentsPtr extents(input);
                LibFoo::calculateRND(RawPtr()(input), output, T_numDims, extents());
            }
        };

        // Complex 1D
        template< class DUMMY >
        struct ExecutionPolicy< 1, true, true, DUMMY >
        {
            void operator()(Input& input, Output output){
                LibFoo::calculateC1D(RawPtr()(input), output, Extents(input)[0]);
            }
        };

        // Complex ND
        template< unsigned T_numDims, class DUMMY >
        struct ExecutionPolicy< T_numDims, true, true, DUMMY >
        {
            void operator()(Input& input, Output output){
                ExtentsPtr extents(input);
                LibFoo::calculateCND(RawPtr()(input), output, T_numDims, extents());
            }
        };

        // Complex(SoA) 1D
        template< class DUMMY >
        struct ExecutionPolicy< 1, true, false, DUMMY >
        {
            void operator()(Input& input, Output output){
                auto ptr = RawPtr()(input);
                LibFoo::calculateC1D(ptr.first, ptr.second, output, Extents(input)[0]);
            }
        };

        // Complex(SoA) ND
        template< unsigned T_numDims, class DUMMY >
        struct ExecutionPolicy< T_numDims, true, false, DUMMY >
        {
            void operator()(Input& input, Output output){
                ExtentsPtr extents(input);
                auto ptr = RawPtr()(input);
                LibFoo::calculateCND(ptr.first, ptr.second, output, T_numDims, extents());
            }
        };

    public:
        using type = ExecutionPolicy< numDims, isComplex, isAoS >;
    };

    template< class T_Input, class Output >
    void
    useIntensityCalculator(T_Input& input, Output output){
        typename IntensityCalculator<T_Input>::type calculator;
        calculator(input, output);
    }

}  // namespace algorithm
}  // namespace foobar
