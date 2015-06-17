#pragma once

#include "foobar/c++14_types.hpp"
#include "foobar/AutoDetect.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/policies/GetStrides.hpp"
#include "foobar/types/RealValues.hpp"
#include "foobar/types/ComplexAoSValues.hpp"
#include "foobar/types/ComplexSoAValues.hpp"

namespace foobar {

    namespace detail {

        template<bool T_isAoS>
        struct SetPointer
        {
            template< typename T_Memory, typename T_Ptr >
            static void
            setPointer(T_Memory*& memory, T_Ptr* ptr)
            {
                memory = reinterpret_cast<T_Memory*>(ptr);
            }
        };

        template<>
        struct SetPointer<false>
        {
            template< typename T_Memory, typename T_Ptr >
            static void
            setPointer(T_Memory& memory, T_Ptr ptr)
            {
                memory.real = reinterpret_cast<decltype(memory.real)>(ptr.first);
                memory.imag = reinterpret_cast<decltype(memory.imag)>(ptr.second);
            }
        };

    }  // namespace detail

    template< typename T_Base >
    class FFT_DataWrapper
    {
    public:
        static constexpr unsigned numDims = traits::NumDims< T_Base >::value;
        static constexpr bool isComplex = traits::IsComplex< T_Base >::value;
        static constexpr bool isAoS = traits::IsAoS< T_Base >::value;
        static constexpr bool isStrided = traits::IsStrided< T_Base >::value;

        using RawPtr = policies::GetRawPtr< T_Base >;
        using RawPtrType = typename std::result_of<RawPtr(T_Base&)>::type; // Type returned by RawPtr(base)
        using Extents = policies::GetExtents< T_Base >;
        using Strides = policies::GetStrides< T_Base >;
        // Precision (double, float...) is the base type of RawPtrType (which is a pointer)
        // For Complex-SoA values RawPtrType is a std::pair of pointers
        using Precision = typename std::remove_pointer<
                              typename std::conditional_t<
                                  isAoS,
                                  std::pair<RawPtrType, RawPtrType>,
                                  RawPtrType
                              >::first_type
                          >::type;

        using Memory = std::conditional_t<
                           isComplex,
                           std::conditional_t<
                               isAoS,
                               types::ComplexAoSValues<Precision>,
                               types::ComplexSoAValues<Precision>
                           >,
                           types::RealValues<Precision>
                       >;

    private:
        T_Base& base_;
        std::array<unsigned, numDims> extents_;

    public:

        FFT_DataWrapper(T_Base& data): base_(data){
            Extents extents(base_);
            for(unsigned i=0; i<numDims; ++i)
                extents_[i] = extents[i];
        }

        Memory
        getData()
        {
            Memory ptr;
            detail::SetPointer<isAoS>::setPointer(ptr, RawPtr()(base_));
            return ptr;
        }

        Extents
        getExtents()
        {
            return Extents(base_);
        }

        unsigned*
        getExtentsPtr()
        {
            return extents_.data();
        }
    };

}  // namespace foobar
