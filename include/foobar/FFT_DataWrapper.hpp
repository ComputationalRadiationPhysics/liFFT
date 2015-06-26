#pragma once

#include "foobar/AutoDetect.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetStrides.hpp"
#include "foobar/types/RealValues.hpp"
#include "foobar/types/ComplexAoSValues.hpp"
#include "foobar/types/ComplexSoAValues.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/c++14_types.hpp"
#include "foobar/FFT_Memory.hpp"
#include <type_traits>
#include <functional>

namespace foobar {

    /**
     * Wrapper for the data
     */
    template<
        class T_FFT_Def,
        class T_IsInput,
        typename T_Base,
        typename T_BaseAccessor = traits::DefaultAccessor_t<T_Base>
    >
    class FFT_DataWrapper
    {
    public:
        using FFT_Def =  T_FFT_Def;
        static constexpr bool isInput = T_IsInput::value;
        using Base = T_Base;
        using BaseAccessor = T_BaseAccessor;

        static constexpr unsigned numDims = traits::NumDims< Base >::value;
        static_assert(numDims == FFT_Def::numDims, "Wrong number of dimensions");
        using IdxType = types::Vec<numDims>;
        using AccRefType = std::result_of_t<BaseAccessor(const IdxType&, Base&)>;
        using AccType = typename std::remove_reference<AccRefType>::type;
        static constexpr bool isComplex = traits::IsComplex< AccType >::value;
        static_assert( (isInput && isComplex == FFT_Def::isComplexInput) ||
                       (!isInput && isComplex == FFT_Def::isComplexOutput),
                       "Wrong element type (complex/real) for this FFT" );

        using Extents = std::array<unsigned, numDims>;
        // Precision (double, float...) is the base type of RawPtrType (which is a pointer)
        // For Complex-SoA values RawPtrType is a std::pair of pointers
        using PrecisionType = typename traits::IntegralType<AccType>::type;
        static_assert(std::is_same<PrecisionType, typename FFT_Def::PrecisionType>::value, "Wrong precision type");

        static constexpr bool needOwnMemoryPtr = !std::is_reference<AccRefType>::value || std::is_const<AccRefType>::value;
        //static constexpr bool needOwnMemory = needOwnMemoryPtr && !FFT_Def::isInplace;
        static constexpr bool isAoS = needOwnMemoryPtr || traits::IsAoS< Base >::value;
        static constexpr bool isStrided = !needOwnMemoryPtr && traits::IsStrided< Base >::value;

        using Memory_t = std::conditional_t<
                           isComplex,
                           std::conditional_t<
                               isAoS,
                               types::ComplexAoSValues<PrecisionType, needOwnMemoryPtr>,
                               types::ComplexSoAValues<PrecisionType, needOwnMemoryPtr>
                           >,
                           types::RealValues<PrecisionType, needOwnMemoryPtr>
                       >;
        using Memory = detail::FFT_Memory< Memory_t, needOwnMemoryPtr >;

    private:
        Base& base_;
        BaseAccessor acc_;
        Extents extents_;
        Memory memory_;
    public:

        FFT_DataWrapper(Base& data): FFT_DataWrapper(data, BaseAccessor()){}

        FFT_DataWrapper(Base& data, BaseAccessor acc): base_(data), acc_(std::move(acc)){
            policies::GetExtents<Base> extents(base_);
            for(unsigned i=0; i<numDims; ++i)
                extents_[i] = extents[i];
            memory_.init(extents_);
        }

        auto
        getDataPtr()
        -> decltype(memory_.getPtr(base_, acc_))
        {
            return memory_.getPtr(base_, acc_);
        }

        const Extents&
        getExtents()
        {
            return extents_;
        }

        unsigned*
        getExtentsPtr()
        {
            return extents_.data();
        }

        unsigned
        getNumElements() const
        {
            return std::accumulate(std::begin(extents_), std::end(extents_), 1u, std::multiplies<unsigned>());
        }

        void
        preProcess()
        {
            if(isInput)
                memory_.copyFrom(base_, acc_);
        }

        void
        postProcess()
        {
            if(!isInput)
                memory_.copyTo(base_, acc_);
        }
    };

    template< class T_FFT_Def, typename T_Base, typename T_BaseAccessor = traits::DefaultAccessor_t<T_Base> >
    using FFT_InputDataWrapper = FFT_DataWrapper< T_FFT_Def, std::true_type, T_Base, T_BaseAccessor >;

    template< class T_FFT_Def, typename T_Base, typename T_BaseAccessor = traits::DefaultAccessor_t<T_Base> >
    using FFT_OutputDataWrapper = FFT_DataWrapper< T_FFT_Def, std::false_type, T_Base, T_BaseAccessor >;

}  // namespace foobar
