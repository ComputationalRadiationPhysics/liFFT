#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/accessors/DataContainerAccessor.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetNumElements.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/mem/RealValues.hpp"
#include "foobar/mem/ComplexAoSValues.hpp"
#include "foobar/void_t.hpp"
#include "foobar/c++14_types.hpp"

#include <cassert>

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
            static constexpr bool isFlatMemory = T_isFlatMemory;

            using Accessor = accessors::DataContainerAccessor<T_isFlatMemory>;
            using IdxType = types::Vec< numDims >;

            friend struct accessors::DataContainerAccessor<isFlatMemory>;
            friend struct policies::GetExtents<DataContainer>;

            /**
             * Creates an uninitialized data container
             */
            DataContainer() = default;
            /**
             * Creates a data container with the specified extents and allocates the memory
             * Note: If the underlying memory does not support automatic destruction
             * you must use freeData() to free the memory
             *
             * @param extents Extents of the container
             */
            DataContainer(const IdxType& extents)
            {
                allocData(extents);
            }

            /**
             * Creates a data container with the specified extents
             *
             * @param extents Extents of the container
             */
            DataContainer(const Memory& data, const IdxType& extents): data(data), extents(extents)
            {}

            /**
             * Creates a data container with the specified extents
             *
             * @param extents Extents of the container
             */
            DataContainer(Memory&& data, const IdxType& extents): data(std::move(data)), extents(extents)
            {}

            template< typename T_Idx >
            std::result_of_t< Accessor(T_Idx&, DataContainer&) >
            operator()(T_Idx& idx)
            {
                assert(policies::checkSizes(idx, extents));
                return Accessor()(idx, *this);
            }

            template< typename T_Idx >
            std::result_of_t< Accessor(T_Idx&, const DataContainer&) >
            operator()(T_Idx& idx) const
            {
                assert(policies::checkSizes(idx, extents));
                return Accessor()(idx, *this);
            }

            void
            setData(const IdxType& extents, const Memory& data)
            {
                this->extents = extents;
                this->data = data;
            }

            void
            setData(const IdxType& extents, Memory&& data)
            {
                this->extents = extents;
                this->data = std::move(data);
            }

            /**
              * Allocates data in the underlying memory if it supports that
              * Note: If the underlying memory does not support automatic destruction
              * you must use freeData() to free the memory
              */
             void
             allocData(const IdxType& extents)
             {
                 this->extents = extents;
                 data.allocData(policies::getNumElements(*this, true));
             }

             /**
               * Frees data in the underlying memory if it supports that
               */
             void
             freeData()
             {
                 data.freeData();
                 extents = IdxType::all(0);
             }

             template<class T>
             struct GetData
             {
                 static auto
                 getData(Memory& data)
                 -> decltype(data.getData())
                 {
                     return data.getData();
                 }
             };

             template<class T>
             struct GetData<T*>
             {
                 static Memory
                 getData(Memory data)
                 {
                     return data;
                 }
             };

             /**
              * Tries to return a pointer to the data
              * That is if Memory is a pointer type, data is returned, otherwise data.getData() is returned
              * @return Pointer to memory
              */
             std::result_of_t< decltype(&GetData<Memory>::getData)(Memory&) >
             getData()
             {
                 return GetData<Memory>::getData(data);
             }

             const IdxType&
             getExtents() const
             {
                 return extents;
             }

        protected:
             Memory data;
             IdxType extents;
        };

        template< unsigned T_numDims, class T_Memory, class T_BaseAccessor, bool T_isFlatMemory >
        struct DataContainer< T_numDims, T_Memory, T_BaseAccessor, T_isFlatMemory, true >: DataContainer< T_numDims, T_Memory, T_BaseAccessor, T_isFlatMemory, false >
        {
            using Parent = DataContainer< T_numDims, T_Memory, T_BaseAccessor, T_isFlatMemory, false >;

            using IdxType = typename Parent::IdxType;
            using Memory = typename Parent::Memory;

            DataContainer(): Parent(){}
            /**
             * Creates a data container with the specified extents and strides and optionally allocates the memory
             * Note: If the underlying memory does not support automatic destruction
             * you must use freeData() to free the memory
             *
             * @param extents Extents of the container
             * @param strides Strides of the container
             */
            DataContainer(const IdxType& extents, const IdxType& strides): Parent(extents), strides(strides)
            {}

            /**
             * Creates a data container with the specified extents
             *
             * @param extents Extents of the container
             */
            DataContainer(const Memory& data, const IdxType& extents, const IdxType& strides): Parent(data, extents), strides(strides)
            {}

            /**
             * Creates a data container with the specified extents
             *
             * @param extents Extents of the container
             */
            DataContainer(Memory&& data, const IdxType& extents, const IdxType& strides): Parent(std::move(data), extents), strides(strides)
            {}


        private:
            IdxType strides;

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
