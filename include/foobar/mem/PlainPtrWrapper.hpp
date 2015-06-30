#pragma once

#include "foobar/policies/flattenIdx.hpp"
#include "foobar/mem/DataContainer.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/types/Real.hpp"
#include "foobar/types/Complex.hpp"
#include "foobar/util.hpp"
#include "foobar/policies/SafePtrCast.hpp"
#include "foobar/policies/ArrayAccessor.hpp"

namespace foobar {
namespace mem {

    /**
     * Wrapper to use plain pointers in the framework
     * This adds dimensions, extents and strides
     */
    template< class T_NumDims, typename T_Pointer, class T_IsStrided >
    class PlainPtrWrapper: protected DataContainer< T_NumDims::value, T_Pointer*, void, T_IsStrided::value>
    {
    public:
        static constexpr unsigned numDims = T_NumDims::value;
        using Pointer = T_Pointer;
        using IntegralType = typename traits::IntegralType<Pointer>::type;
        static constexpr bool isStrided = T_IsStrided::value;

        using Accessor = policies::ArrayAccessor< true >;

        using Ref = Pointer&;
        using ConstRef = const Pointer&;

        PlainPtrWrapper(Pointer* ptr, types::Vec<numDims> extents): data(ptr), extents(extents)
        {
            static_assert(!isStrided, "You need to specify the strides!");
        }

        PlainPtrWrapper(IntegralType* ptr, types::Vec<numDims> extents):
            PlainPtrWrapper(policies::safe_ptr_cast<Pointer*>(ptr), extents)
        {}

        PlainPtrWrapper(Pointer* ptr, types::Vec<numDims> extents, types::Vec<numDims> strides): data(ptr), extents(extents), strides(strides)
        {
            static_assert(isStrided, "You cannot specify strides!!");
        }

        PlainPtrWrapper(IntegralType* ptr, types::Vec<numDims> extents, types::Vec<numDims> strides):
            PlainPtrWrapper(policies::safe_ptr_cast<Pointer*>(ptr), extents, strides)
        {}

        template< class T_Index >
        Ref
        operator()(T_Index&& idx)
        {
            unsigned flatIdx = policies::flattenIdx(idx, *this);
            return data[flatIdx];
        }

        template< class T_Index >
        ConstRef
        operator()(T_Index&& idx) const
        {
            unsigned flatIdx = policies::flattenIdx(idx, *this);
            return data[flatIdx];
        }
    };

    /**
     * Evaluates to Real<T> or Complex<T> depending on the template argument
     */
    template< bool T_isComplex, typename T >
    struct RealOrComplex: std::conditional< T_isComplex, types::Real<T>, types::Complex<T>>
    {
        static_assert(std::is_floating_point<T>::value, "Only floating point values are allowed!");
    };
    template< bool T_isComplex, typename T >
    using RealOrComplex_t = typename RealOrComplex< T_isComplex, T >::type;


    // Following are the convenience functions for wrapping pointers

    template< bool T_isComplex, typename T >
    PlainPtrWrapper<UnsignedConst<1>, RealOrComplex_t< T_isComplex, T >, std::false_type>
    wrapPtr(T* ptr, unsigned size)
    {
        return PlainPtrWrapper<UnsignedConst<1>, RealOrComplex_t< T_isComplex, T >, std::false_type>(ptr, types::Idx1D(size));
    }

    template< bool T_isComplex, typename T >
    PlainPtrWrapper<UnsignedConst<1>, RealOrComplex_t< T_isComplex, T >, std::true_type>
    wrapPtrStrided(T* ptr, unsigned size, unsigned stride)
    {
        return PlainPtrWrapper<UnsignedConst<1>, RealOrComplex_t< T_isComplex, T >, std::true_type>(ptr, types::Idx1D(size), types::Idx1D(stride));
    }

    template< bool T_isComplex, typename T >
    PlainPtrWrapper<UnsignedConst<2>, RealOrComplex_t< T_isComplex, T >, std::false_type>
    wrapPtr(T* ptr, unsigned sizeY, unsigned sizeX)
    {
        return PlainPtrWrapper<UnsignedConst<2>, RealOrComplex_t< T_isComplex, T >, std::false_type>(ptr, types::Idx2D(sizeY, sizeX));
    }

    template< bool T_isComplex, typename T >
    PlainPtrWrapper<UnsignedConst<2>, RealOrComplex_t< T_isComplex, T >, std::true_type>
    wrapPtrStrided(T* ptr, unsigned sizeY, unsigned sizeX, unsigned strideY, unsigned strideX)
    {
        return PlainPtrWrapper<UnsignedConst<2>, RealOrComplex_t< T_isComplex, T >, std::true_type>(ptr, types::Idx2D(sizeY, sizeX), types::Idx2D(strideY, strideX));
    }

    template< bool T_isComplex, typename T >
    PlainPtrWrapper<UnsignedConst<3>, RealOrComplex_t< T_isComplex, T >, std::false_type>
    wrapPtr(T* ptr, unsigned sizeZ, unsigned sizeY, unsigned sizeX)
    {
        return PlainPtrWrapper<UnsignedConst<3>, RealOrComplex_t< T_isComplex, T >, std::false_type>(ptr, types::Idx3D(sizeZ, sizeY, sizeX));
    }

    template< bool T_isComplex, typename T >
    PlainPtrWrapper<UnsignedConst<3>, RealOrComplex_t< T_isComplex, T >, std::true_type>
    wrapPtrStrided(T* ptr, unsigned sizeZ, unsigned sizeY, unsigned sizeX, unsigned strideZ, unsigned strideY, unsigned strideX)
    {
        return PlainPtrWrapper<UnsignedConst<3>, RealOrComplex_t< T_isComplex, T >, std::true_type>(ptr, types::Idx3D(sizeZ, sizeY, sizeX), types::Idx3D(strideZ, strideY, strideX));
    }

}  // namespace mem
}  // namespace foobar
