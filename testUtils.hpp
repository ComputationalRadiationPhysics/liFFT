#pragma once

#include "testDefines.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/policies/Loop.hpp"
#include <iosfwd>
#include <cmath>

namespace foobarTest {

    /**
     * Initializes the test environment and prepares the base line FFTs to compare against
     */
    void init();
    /**
     * Frees all resources used in the test environment
     */
    void finalize();
    /**
     * Executes the base tests and outputs PDFs visualizing the results
     */
    void visualizeBase();

    /**
     * Executes the FFT on the base Input for R2C
     */
    void execBaseR2C();
    /**
     * Executes the FFT on the base Input for C2C
     */
    void execBaseC2C();

    /**
     * Maximum error detected during a compare run
     */
    struct CmpError{
        double maxAbsDiff = 0;
        double maxRelDiff = 0;

        CmpError(): maxAbsDiff(0), maxRelDiff(0)
        {}
        CmpError(double allowedAbsDiff, double allowedRelDiff): maxAbsDiff(allowedAbsDiff), maxRelDiff(allowedRelDiff)
        {}

        friend inline std::ostream& operator<<(std::ostream& stream, const CmpError& e){
            stream << "Max AbsDiff = " << e.maxAbsDiff  << " Max RelDiff = " << e.maxRelDiff;
            return stream;
        }
    };


    /**
     * "Functor" used for comparing multidimensional containers of Real or Complex data
     */
    struct CompareFunc
    {
        bool ok = true;
        CmpError e;
        CmpError allowed_;

        CompareFunc(CmpError allowed): allowed_(allowed){}

        template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
        void handleLoopPre(T_Args&&...){}
        template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
        void handleLoopPost(T_Args&&...){}

        template<class T, class U>
        std::enable_if_t< foobar::traits::IsComplex<T>::value, bool >
        compare(const T& expected, const U& is)
        {
            return compare(expected.real, is.real) && compare(expected.imag, is.imag);
        }

        template<class T, class U>
        std::enable_if_t< !foobar::traits::IsComplex<T>::value, bool >
        compare(const T& expected, const U& is)
        {
            if(expected == is)
                return true;
            auto absDiff = std::abs(expected-is);
            if(absDiff <= allowed_.maxAbsDiff)
                return true;
            auto relDiff = std::abs(absDiff / expected);
            if(relDiff <= allowed_.maxRelDiff)
                return true;
            if(absDiff > e.maxAbsDiff)
                e.maxAbsDiff = absDiff;
            if(relDiff > e.maxRelDiff)
                e.maxRelDiff = relDiff;
            return false;
        }

        template<
            unsigned T_curDim,
            class T_Index,
            class T_Src,
            class T_SrcAccessor,
            class T_Dst,
            class T_DstAccessor
            >
        void
        handleInnerLoop(const T_Index& idx, const T_Src& expected, T_SrcAccessor&& accExp, T_Dst& is, T_DstAccessor&& accIs)
        {
            if(!compare(accExp(idx, expected), accIs(idx, is)))
                ok = false;
        }
    };

    /**
     * Compares 2 multidimensional containers
     *
     * @param expected   Container with expected values
     * @param is         Container with actual values
     * @param allowedErr Maximum allowed error
     * @param expAcc     Accessor for expected container [DefaultAccessor used]
     * @param isAcc      Accessor for actual container [DefaultAccessor used]
     * @return Pair: 1: bool OK, 2: Maximum errors detected
     */
    template< class T, class U, class T_AccessorT = foobar::traits::DefaultAccessor_t<T>, class T_AccessorU = foobar::traits::DefaultAccessor_t<U> >
    std::pair< bool, CmpError >
    compare(const T& expected, const U& is, CmpError allowedErr = CmpError(5e-5, 5e-5), const T_AccessorT& expAcc = T_AccessorT(), const T_AccessorU& isAcc = T_AccessorU())
    {
        CompareFunc result(allowedErr);
        foobar::policies::loop(expected, result, expAcc, is, isAcc);
        return std::make_pair(result.ok, result.e);
    }

}  // namespace foobarTest
