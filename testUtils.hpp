#pragma once

#include "testDefines.hpp"
#include "foobar/traits/DefaultAccessor.hpp"

void initTest();
void visualizeBaseTest();
void finalizeTest();

void testExecBaseR2C();
void testExecBaseC2C();

template< class T, class U, class T_AccessorT = foobar::traits::DefaultAccessor_t<T>, class T_AccessorU = foobar::traits::DefaultAccessor_t<U> >
bool
compare(const T& left, const U& right, const T_AccessorT& leftAcc, const T_AccessorU& rightAcc)
{
    return true;
}
