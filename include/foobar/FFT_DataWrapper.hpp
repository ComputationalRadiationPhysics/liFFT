#pragma once

#include "foobar/AutoDetect.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetStrides.hpp"
#include "foobar/mem/RealValues.hpp"
#include "foobar/mem/ComplexAoSValues.hpp"
#include "foobar/mem/ComplexSoAValues.hpp"
#include "foobar/mem/DataContainer.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/c++14_types.hpp"
#include "foobar/FFT_Memory.hpp"
#include <type_traits>
#include <functional>

namespace foobar {

    // Fwd decl
    template< class T1, class T2, class T3, bool t>
    class FFT;

    namespace detail {

        template< class T_FFT_DataWrapper, bool T_isHalfData = T_FFT_DataWrapper::isHalfData >
        struct GetFullData
        {
            types::SymmetricWrapper<T_FFT_DataWrapper>
            operator()(T_FFT_DataWrapper& data) const
            {
                return types::makeSymmetricWrapper(data, data.realExtents_[data.numDims-1]);
            }
        };

        template<class T_FFT_DataWrapper>
        struct GetFullData< T_FFT_DataWrapper, false >
        {
            T_FFT_DataWrapper
            operator()(T_FFT_DataWrapper& data) const
            {
                return data;
            }
        };

    }  // namespace detail

    /**
     * Gets the complete dataset for a dataset of a C2R/R2C FFT
     * Returns the same data for the real types or any wrapper used in a C2C FFT
     * MUST be called AFTER the FFT was executed!
     *
     * @param data FFT_DataWrapper instance
     * @return Complete dataset
     */
    template< class T_FFT_DataWrapper >
    std::result_of_t<detail::GetFullData<T_FFT_DataWrapper>(T_FFT_DataWrapper&)>
    getFullData(T_FFT_DataWrapper& data)
    {
        return detail::GetFullData<T_FFT_DataWrapper>()(data);
    }

    /**
     * Wrapper for the data
     */
    template<
        class T_FFT_Def,
        class T_IsInput,
        typename T_Base,
        class T_HasInstance = std::false_type,
        typename T_BaseAccessor = traits::DefaultAccessor_t<T_Base>
    >
    class FFT_DataWrapper
    {
    public:
        using FFT_Def =  T_FFT_Def;
        static constexpr bool isInput = T_IsInput::value;
        using Base = T_Base;
        using BaseAccessor = T_BaseAccessor;
        static constexpr bool hasInstance = T_HasInstance::value;

        static_assert(!FFT_Def::isInplace, "Use the FFT_InplaceDataWrapper for inplace transforms");
        static constexpr unsigned numDims = traits::NumDims< Base >::value;
        static_assert(numDims == FFT_Def::numDims, "Wrong number of dimensions");
        using IdxType = types::Vec<numDims>;
        using AccRefType = std::result_of_t<BaseAccessor(const IdxType&, Base&)>;
        using AccType = typename std::remove_reference<AccRefType>::type;
        static constexpr bool isComplex = traits::IsComplex< AccType >::value;
        static_assert( (isInput && (isComplex || !FFT_Def::isComplexInput)) ||
                       (!isInput && (isComplex || !FFT_Def::isComplexOutput)),
                       "Wrong element type for this FFT: Expected complex, got real" );
        static_assert( (isInput && (!isComplex || FFT_Def::isComplexInput)) ||
                       (!isInput && (!isComplex || FFT_Def::isComplexOutput)),
                       "Wrong element type for this FFT: Expected real, got complex" );

        using Extents = IdxType;
        // Precision (double, float...) is the base type of RawPtrType (which is a pointer)
        // For Complex-SoA values RawPtrType is a std::pair of pointers
        using PrecisionType = typename traits::IntegralType<AccType>::type;
        static_assert(std::is_same<PrecisionType, typename FFT_Def::PrecisionType>::value, "Wrong precision type");

        static constexpr bool needOwnMemoryPtr = !std::is_reference<AccRefType>::value || std::is_const<AccRefType>::value;
        static constexpr bool isAoS = needOwnMemoryPtr || traits::IsAoS< Base >::value;
        static constexpr bool isStrided = !needOwnMemoryPtr && traits::IsStrided< Base >::value;

        using Memory_t = mem::DataContainer<
                             numDims,
                             std::conditional_t<
                                 isComplex,
                                 std::conditional_t<
                                     isAoS,
                                     mem::ComplexAoSValues<PrecisionType, needOwnMemoryPtr>,
                                     mem::ComplexSoAValues<PrecisionType, needOwnMemoryPtr>
                                 >,
                                 mem::RealValues<PrecisionType, needOwnMemoryPtr>
                             >
                         >;
        using Memory = detail::FFT_Memory< Memory_t, needOwnMemoryPtr >;
        using MemoryFallback = detail::FFT_Memory< Memory_t, true >;

        using Accessor = accessors::ArrayAccessor<true>;

        using RefType = typename std::add_lvalue_reference<Base>::type;
        using InstanceType = std::conditional_t< hasInstance, Base, RefType >;
        using ParamType = typename std::conditional_t< hasInstance, std::add_rvalue_reference<Base>, std::add_lvalue_reference<Base> >::type;
        static constexpr bool isHalfData = (FFT_Def::kind == FFT_Kind::Complex2Real && isInput) ||
                                           (FFT_Def::kind == FFT_Kind::Real2Complex && !isInput);
        friend class detail::GetFullData<FFT_DataWrapper>;
    private:
        InstanceType base_;
        BaseAccessor acc_;
        Extents extents_, realExtents_;
        Memory memory_;
        std::unique_ptr<MemoryFallback> memFallback_;

        void
        setRealExtents(const Extents& extents)
        {
            realExtents_ = extents;
        }
        template< class T1, class T2, class T3, bool t>
        friend class FFT;
    public:

        FFT_DataWrapper(ParamType data):
            FFT_DataWrapper(static_cast<ParamType>(data), BaseAccessor()){}

        FFT_DataWrapper(ParamType data, BaseAccessor acc):
            base_(static_cast<ParamType>(data)), acc_(std::move(acc))
        {
            policies::GetExtents<Base> extents(base_);
            for(unsigned i=0; i<numDims; ++i)
                extents_[i] = extents[i];
            memory_.init(extents_);
            if(memory_.checkPtr(base_, acc_))
                memFallback_ = nullptr;
            else
            {
                memFallback_.reset(new MemoryFallback());
                memFallback_->init(extents_);
            }
        }

        /**
         * Returns an element from the base class
         * @param idx Index
         * @return    Element
         */
        std::result_of_t< BaseAccessor(const IdxType&, Base&) >
        operator()(const IdxType& idx)
        {
            return acc_(idx, base_);
        }

        /**
         * Returns an element from the base class
         * @param idx Index
         * @return    Element
         */
        std::result_of_t< BaseAccessor(const IdxType&, const Base&) >
        operator()(const IdxType& idx) const
        {
            return acc_(idx, const_cast<const Base&>(base_));
        }

        /**
         * Returns a reference to the base class
         * @return Reference to base data
         */
        RefType
        getBase()
        {
            return base_;
        }

        /**
         * Returns the pointer to the internal data
         * This might be memory allocated by this class or the memory from the base type
         *
         * @return Complex* or Real*
         */
        auto
        getDataPtr()
        -> decltype(memory_.getPtr(base_, acc_))
        {
            if(memFallback_)
                return memFallback_->getPtr(base_, acc_);
            else
                return memory_.getPtr(base_, acc_);
        }

        /**
         * Returns the extents object
         * Supports iterators
         *
         * @return Extents object
         */
        const Extents&
        getExtents() const
        {
            return extents_;
        }

        /**
         * Returns the plain ptr to the extents
         */
        const unsigned*
        getExtentsPtr() const
        {
            return extents_.data();
        }

        /**
         * Returns the number of actual elements (ignoring strides)
         */
        unsigned
        getNumElements() const
        {
            return std::accumulate(extents_.cbegin(), extents_.cend(), 1u, std::multiplies<unsigned>());
        }

        /**
         * Internal method. Called before each FFT for input
         * Copies data to internal memory if required
         */
        void
        preProcess()
        {
            if(isInput)
            {
                if(memFallback_)
                    memFallback_->copyFrom(base_, acc_);
                else
                    memory_.copyFrom(base_, acc_);
            }
        }

        /**
         * Internal method. Called after each FFT for output wrappers
         * Copies data from internal memory if required
         */
        void
        postProcess()
        {
            if(!isInput)
            {
                if(memFallback_)
                    memFallback_->copyTo(base_, acc_);
                else
                    memory_.copyTo(base_, acc_);
            }
        }
    };

    template< class T_FFT_Def, typename T_Base, class T_HasInstance = std::false_type, typename T_BaseAccessor = traits::DefaultAccessor_t<T_Base> >
    using FFT_InputDataWrapper = FFT_DataWrapper< T_FFT_Def, std::true_type, T_Base, T_HasInstance, T_BaseAccessor >;

    template< class T_FFT_Def, typename T_Base, class T_HasInstance = std::false_type, typename T_BaseAccessor = traits::DefaultAccessor_t<T_Base> >
    using FFT_OutputDataWrapper = FFT_DataWrapper< T_FFT_Def, std::false_type, T_Base, T_HasInstance, T_BaseAccessor >;

    namespace policies {

        template< class... T >
        struct GetExtents< FFT_DataWrapper< T... > >
        {
            using Data = FFT_DataWrapper< T... >;
            GetExtents(const Data& data): data_(data){}

            unsigned operator[](unsigned dimIdx) const
            {
                return data_.getExtents()[dimIdx];
            }
        private:
            const Data& data_;
        };

    }  // namespace policies

}  // namespace foobar
