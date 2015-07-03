#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/policies/DataContainerAccessor.hpp"
#include "foobar/policies/GetNumElements.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/mem/RealValues.hpp"
#include "foobar/mem/ComplexAoSValues.hpp"
#include "foobar/void_t.hpp"

namespace foobar {

    namespace mem {

        /**
         * Container used to store data with its meta-data
         */
        template< unsigned T_numDims, class T_Memory, class T_BaseAccessor = traits::DefaultAccessor_t<T_Memory>, bool T_isFlatMemory = true, bool T_isStrided=false >
        struct DataContainer
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
            using BaseAccessor = T_BaseAccessor;
            static constexpr bool isStrided = T_isStrided;

            using Accessor = policies::DataContainerAccessor<T_isFlatMemory>;
            using IdxType = types::Vec< numDims >;

            IdxType extents;
            Memory data;

            /**
             * Creates an uninitialized data container
             */
            DataContainer() = default;
            /**
             * Creates a data container with the specified extents and optionally allocates the memory
             * Note: If the underlying memory does not support automatic destruction
             * you must use freeData() to free the memory
             *
             * @param extents Extents of the container
             */
            DataContainer(const IdxType& extents, bool alloc = true): extents(extents)
            {
                if(alloc)
                    allocData();
            }

            /**
              * Allocates data in the underlying memory if it supports that
              * Note: If the underlying memory does not support automatic destruction
              * you must use freeData() to free the memory
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
            using Parent = DataContainer< T_numDims, T_Memory, T_BaseAccessor, T_isFlatMemory, false >;

            using IdxType = typename Parent::IdxType;

            IdxType strides;

            DataContainer(): Parent(){}
            /**
             * Creates a data container with the specified extents and strides and optionally allocates the memory
             * Note: If the underlying memory does not support automatic destruction
             * you must use freeData() to free the memory
             *
             * @param extents Extents of the container
             * @param strides Strides of the container
             */
            DataContainer(const IdxType& extents, const IdxType& strides = IdxType::all(0), bool alloc = true): Parent(extents, false), strides(strides)
            {
                if(alloc)
                    this->allocData();
            }
        };

        /**
         * A container storing real data with automatic memory management
         */
        template< unsigned T_numDims, typename T_Precision, bool T_isStrided = false >
        using RealContainer = DataContainer< T_numDims, RealValues<T_Precision>, typename RealValues<T_Precision>::Accessor, true, false >;

        /**
         * A container storing complex data with automatic memory management
         */
        template< unsigned T_numDims, typename T_Precision, bool T_isStrided = false >
        using ComplexContainer = DataContainer< T_numDims, ComplexAoSValues<T_Precision>, typename ComplexAoSValues<T_Precision>::Accessor, true, false >;

    }  // namespace mem

    namespace traits {

        template< class T >
        struct IntegralTypeImpl< T, void_t< typename T::Memory > >: IntegralType< typename T::Memory >{};

        template< class T >
        struct IsComplex<T, void_t< typename T::Memory > >: IsComplex< typename T::Memory >{};

        template< class T >
        struct IsAoS<T, void_t< typename T::Memory > >: IsAoS< typename T::Memory >{};

    }  // namespace traits

}  // namespace foobar
