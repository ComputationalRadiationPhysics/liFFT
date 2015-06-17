#pragma once

template <typename Assertion>
struct AssertValue
{
    static bool const value = Assertion::value;
    static_assert(value, "Assertion failed <see above for more information>");
};

template <class T, class M>
M get_member_type(M T:: *);

#define GET_TYPE_OF(mem) decltype(get_member_type(&mem))
