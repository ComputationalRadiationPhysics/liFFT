#pragma once

template <typename Assertion>
struct AssertValue
{
    static bool const value = Assertion::value;
    static_assert(value, "Assertion failed <see above for more information>");
};
