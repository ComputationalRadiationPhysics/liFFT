#pragma once

namespace foobar {
namespace policies {

    /**
     * Returns the raw ptr to the underlying data
     * This will be with getData for a single pointer or getRealData/getImagData for ComplexSoA
     * Takes the object in the constructor
     */
    template< typename T_Memory >
    struct GetRawPtr;

}  // namespace policies
}  // namespace foobar
