#pragma once

#include "testDefines.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/types/Complex.hpp"
#include "foobar/types/Real.hpp"
#include "foobar/policies/Loop.hpp"
#include <iostream>

void initTest();
void visualizeBaseTest();
void finalizeTest();

void testExecBaseR2C();
void testExecBaseC2C();

struct CompareTest
{
    bool ok = true;
    double maxDiff = 0;

    template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
    void handleLoopPre(T_Args&&...){}
    template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
    void handleLoopPost(T_Args&&...){}

    template<typename T>
    bool
    compare(const foobar::types::Complex<T>& l, const foobar::types::Complex<T>& r)
    {
        return compare(l.real, r.real) && compare(l.imag, r.imag);
    }

    template<typename T>
    bool
    compare(const foobar::types::Real<T>& l, const foobar::types::Real<T>& r)
    {
        if(abs(l-r) < 1e-8)
            return true;
        else
        {
            auto diff = abs(l-r);
            if(diff > maxDiff)
                maxDiff = diff;
            return false;
        }
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
bool
compare(const T& left, const U& right, const T_AccessorT& leftAcc = T_AccessorT(), const T_AccessorU& rightAcc = T_AccessorU())
{
    CompareTest result;
    foobar::policies::loop(left, result, leftAcc, right, rightAcc);
    if(!result.ok)
        std::cerr << "Max diff = " << result.maxDiff << std::endl;
    return result.ok;
}
