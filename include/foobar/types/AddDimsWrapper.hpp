#pragma once

namespace foobar {
namespace types {

    /**
     * Wrapper used to add dimensionality Information to a class without using traits
     *
     * \tparam T_Base Base class to wrap
     * \tparam T_numDims number of dimensions this class should have
     */
    template< class T_Base, unsigned T_numDims >
    struct AddDimsWrapper: T_Base
    {
        using T_Base::T_Base;
        static constexpr unsigned numDims = T_numDims;
    };

}  // namespace types
}  // namespace foobar
