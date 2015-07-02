#pragma once

#include "testDefines.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/types/Complex.hpp"
#include "foobar/types/Real.hpp"
#include "foobar/policies/Loop.hpp"
#include <iostream>
#include <cmath>

void initTest();
void visualizeBaseTest();
void finalizeTest();

void testExecBaseR2C();
void testExecBaseC2C();

struct CmpError{
    double maxAbsDiff = 0;
    double maxRelDiff = 0;
};

struct CompareTest
{
    bool ok = true;
    CmpError e;

    template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
    void handleLoopPre(T_Args&&...){}
    template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
    void handleLoopPost(T_Args&&...){}

    template<typename T>
    bool
    compare(const foobar::types::Complex<T>& expected, const foobar::types::Complex<T>& is)
    {
        return compare(expected.real, is.real) && compare(expected.imag, is.imag);
    }

    template<typename T>
    bool
    compare(const foobar::types::Real<T>& expected, const foobar::types::Real<T>& is)
    {
        T absDiff = std::abs(expected-is);
        T relDiff = std::abs(absDiff / expected);
        if(absDiff > e.maxAbsDiff)
            e.maxAbsDiff = absDiff;
        if(relDiff > e.maxRelDiff)
            e.maxRelDiff = relDiff;
        if(absDiff < 5e-5 || relDiff < 5e-5)
            return true;
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
    handleInnerLoop(const T_Index& idx, const T_Src& src, T_SrcAccessor&& accSrc, T_Dst& dst, T_DstAccessor&& accDst)
    {
        if(!compare(accSrc(idx, src), accDst(idx, dst)))
            ok = false;
    }
};

template< class T, class U, class T_AccessorT = foobar::traits::DefaultAccessor_t<T>, class T_AccessorU = foobar::traits::DefaultAccessor_t<U> >
CmpError
compare(const T& left, const U& right, const T_AccessorT& leftAcc = T_AccessorT(), const T_AccessorU& rightAcc = T_AccessorU())
{
    CompareTest result;
    foobar::policies::loop(left, result, leftAcc, right, rightAcc);
    if(!result.ok)
        std::cerr << "Max AbsDiff = " << result.e.maxAbsDiff  << " Max RelDiff = " << result.e.maxRelDiff << std::endl;
    return result.e;
}
