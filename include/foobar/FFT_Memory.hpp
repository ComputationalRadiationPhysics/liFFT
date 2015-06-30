#pragma once

#include "foobar/policies/Copy.hpp"
#include "foobar/policies/SafePtrCast.hpp"
#include <algorithm>

namespace foobar {
namespace detail {

    /**
     * Policy for managing the memory of the FFT_DataWrapper
     *
     * \tparam T_Memory Type of the memory to manage (RealValues/Complex*Values)
     * \tparam T_needOwnMemory True if memory should be allocated or if the base memory can be used
     */
    template< typename T_Memory, bool T_needOwnMemory = true >
    struct FFT_Memory
    {
        using Memory = T_Memory;
        using Ptr = std::result_of_t< decltype(&Memory::getData)(Memory) >;

        Memory data_;

        /**
         * Allocates memory such that a matrix with the given extents fits in
         *
         * @param extents Extents of the memory. Needs to support std::begin/end
         */
        template< typename T_Extents >
        void
        init(const T_Extents& extents)
        {
            unsigned numEls = std::accumulate(std::begin(extents), std::end(extents), 1, std::multiplies<unsigned>());
            data_.allocData(numEls);
        }

        /**
         * Returns a pointer to the start of the data
         *
         * @param unused
         * @param unused
         * @return Pointer to the data
         */
        template< class T_Obj, class T_Acc >
        Ptr
        getPtr(T_Obj, T_Acc)
        {
            return data_.getData();
        }

        /**
         * Copies data from the given object into this memory
         *
         * @param obj Object to read from
         * @param acc Accessor to access the elements
         */
        template< class T_Obj, class T_Acc >
        void
        copyFrom(T_Obj& obj, T_Acc&& acc)
        {
            auto copy = policies::makeCopy(typename Memory::Accessor(), std::forward<T_Acc>(acc));
            copy(obj, data_);
        }

        /**
         * Copies data from this memory to the given object
         *
         * @param obj Object to write to
         * @param acc Accessor to access the elements
         */
        template< class T_Obj, class T_Acc >
        void
        copyTo(T_Obj& obj, T_Acc&& acc)
        {
            auto copy = policies::makeCopy(std::forward<T_Acc>(acc), typename Memory::Accessor());
            copy(data_, obj);
        }
    };

    template< typename T_Ptr, bool T_isPlainPtr = std::is_pointer<T_Ptr>::value >
    struct FFT_Memory_GetPtr
    {
        using Ptr = T_Ptr;

        /**
         * Returns a pointer to the start of the data in the given object
         * Enabled if we should return a plain pointer
         *
         * @param obj Object where the data is stored
         * @param acc Accessor to access the elements in @ref obj
         * @return Pointer to the element at acc({0, ... } obj)
         */
        template< class T_Obj, class T_Acc >
        Ptr
        getPtr(T_Obj& obj, T_Acc& acc)
        {
            static constexpr unsigned numDims = traits::NumDims< T_Obj >::value;
            auto idx = types::Vec<numDims>::all(0);
            return policies::safe_ptr_cast<Ptr>(&acc(idx, obj));
        }
    };

    template< typename T_Ptr >
    struct FFT_Memory_GetPtr< T_Ptr, false >
    {
        using Ptr = T_Ptr;

        /**
         * Returns a pointer to the start of the data in the given object
         * Enabled if we should return a pair of pointers where the first element
         * points to the real part and the second to the imaginary
         *
         * @param obj Object where the data is stored
         * @param acc Accessor to access the elements in @ref obj
         * @return std::pair of pointers to the element at acc({0, ... } obj)
         */
        template< class T_Obj, class T_Acc >
        Ptr
        getPtr(T_Obj& obj, T_Acc& acc)
        {
            static constexpr unsigned numDims = traits::NumDims< T_Obj >::value;
            auto idx = types::Vec<numDims>::all(0);
            auto& ref = acc(idx, obj);
            return policies::safe_ptr_cast<Ptr>(std::make_pair(&ref.real, &ref.imag));
        }
    };

    /**
     * Specialization that does not use own memory but forwards everything to the base object
     */
    template< typename T_Memory>
    struct FFT_Memory< T_Memory, false >: public FFT_Memory_GetPtr< std::result_of_t< decltype(&T_Memory::getData)(T_Memory) > >
    {
        using Memory = T_Memory;

        template< typename T_Extents >
        void init(T_Extents){}

        template< class T_Obj, class T_Acc >
        void copyFrom(T_Obj&, T_Acc&){}

        template< class T_Obj, class T_Acc >
        void copyTo(T_Obj&, T_Acc&){}
    };

}  // namespace detail
}  // namespace foobar
