#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/policies/DataContainerAccessor.hpp"
#include "foobar/policies/GetNumElements.hpp"

namespace foobar {

    namespace mem {

        /**
         * Container used to store data with its meta-data
         */
        template< unsigned T_numDims, class T_Memory, class T_BaseAccessor = typename T_Memory::Accessor, bool T_isFlatMemory = true, bool T_isStrided=false >
        struct DataContainer
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
            using BaseAccessor = T_BaseAccessor;
            static constexpr bool isStrided = T_isStrided;

            using Accessor = policies::DataContainerAccessor<T_isFlatMemory>;

            types::Vec< numDims > extents;
            Memory data;

            /**
              * Allocates data in the underlying memory if it supports that
              */
             void
             allocData()
             {
                 data.allocData(policies::getNumElements(*this, true));
             }

             /**
               * Frees data in the underlying memory if it supports that
               */
             void
             freeData()
             {
                 data.freeData();
             }
        };

        template< unsigned T_numDims, class T_Memory, class T_BaseAccessor, bool T_isFlatMemory >
        struct DataContainer< T_numDims, T_Memory, T_BaseAccessor, T_isFlatMemory, true >: DataContainer< T_numDims, T_Memory, T_BaseAccessor, T_isFlatMemory, false >
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
            using BaseAccessor = T_BaseAccessor;
            static constexpr bool isStrided = true;

            types::Vec< numDims > strides;
        };

    }  // namespace types

    namespace traits {

        template< class T >
        struct IntegralTypeImpl< T, void_t< typename T::Memory > >: IntegralType< typename T::Memory >{};

        template< class T >
        struct IsComplex<T, void_t< typename T::Memory > >: IsComplex< typename T::Memory >{};

        template< class T >
        struct IsAoS<T, void_t< typename T::Memory > >: IsAoS< typename T::Memory >{};

    }  // namespace traits

}  // namespace foobar
