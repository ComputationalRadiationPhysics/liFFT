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
#include "foobar/traits/IdentityAccessor.hpp"
#include "foobar/traits/IsDeviceMemory.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/c++14_types.hpp"
#include "foobar/FFT_Memory.hpp"
#include <type_traits>
#include <functional>

namespace foobar {

    // Fwd decl
    template< class T1, class T2, class T3, bool t>
    class FFT;
    template< class > class FFT_InplaceOutput;

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

        struct IInplaceOutput
        {
            virtual void preProcess() = 0;
            virtual void postProcess() = 0;
        protected:
            IInplaceOutput(){}
            virtual ~IInplaceOutput(){}
        };

        template< bool T_isInplace = false >
        class FFT_DataWrapperBase
        {
        protected:
            void preProcess(){}
            void postProcess(){}
            FFT_DataWrapperBase(){}
            ~FFT_DataWrapperBase(){}
        };

        template<>
        class FFT_DataWrapperBase<true>
        {
        protected:
            void preProcess(){
                if(inplaceOutput_)
                    inplaceOutput_->preProcess();
            }
            void postProcess(){
                if(inplaceOutput_)
                    inplaceOutput_->postProcess();
            }
            detail::IInplaceOutput* inplaceOutput_;

            FFT_DataWrapperBase(): inplaceOutput_(nullptr){}
            ~FFT_DataWrapperBase(){}
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
        typename T_BaseAccessor = traits::IdentityAccessor_t<T_Base>
    >
    class FFT_DataWrapper: public detail::FFT_DataWrapperBase<T_FFT_Def::isInplace>
    {
    public:
        using FFT_Def =  T_FFT_Def;
        static constexpr bool isInput = T_IsInput::value;
        using Base = T_Base;
        using BaseAccessor = T_BaseAccessor;
        static constexpr bool hasInstance = T_HasInstance::value;

        using Parent = detail::FFT_DataWrapperBase<T_FFT_Def::isInplace>;

        static_assert(!FFT_Def::isInplace || isInput, "Only the input container must be specified for inplace transforms");
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
        static_assert(!FFT_Def::isInplace || !needOwnMemoryPtr, "Memory used is not writable");
        static constexpr bool isAoS = needOwnMemoryPtr || traits::IsAoS< Base >::value;
        static constexpr bool isStrided = !needOwnMemoryPtr && traits::IsStrided< Base >::value;
        using IsDeviceMemory = traits::IsDeviceMemory< Base >;

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

        using IdentityAccessor = accessors::ArrayAccessor<true>;

        using RefType = typename std::add_lvalue_reference<Base>::type;
        using InstanceType = std::conditional_t< hasInstance, Base, RefType >;
        using ParamType = typename std::conditional_t< hasInstance, std::add_rvalue_reference<Base>, std::add_lvalue_reference<Base> >::type;
        static constexpr bool isHalfData = (FFT_Def::kind == FFT_Kind::Complex2Real && isInput) ||
                                           (FFT_Def::kind == FFT_Kind::Real2Complex && !isInput);
        friend class detail::GetFullData<FFT_DataWrapper>;
        friend class FFT_InplaceOutput<FFT_DataWrapper>;
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
        FFT_DataWrapper(ParamType data, BaseAccessor acc = BaseAccessor()):
            base_(static_cast<ParamType>(data)), acc_(std::move(acc))
        {
            static_assert(!FFT_Def::isInplace || FFT_Def::kind != FFT_Kind::Complex2Real, "No real extents set");

            policies::GetExtents<Base> extents(base_);
            for(unsigned i=0; i<numDims; ++i)
                extents_[i] = extents[i];
            memory_.init(extents_);
            if(memory_.checkPtr(base_, acc_, FFT_Def::isInplace && !isComplex))
                memFallback_ = nullptr;
            else if(FFT_Def::isInplace)
                throw std::runtime_error("Cannot use given memory as the strides/indexing is wrong!");
            else
            {
                memFallback_.reset(new MemoryFallback());
                memFallback_->init(extents_);
            }
        }

        FFT_DataWrapper(ParamType data, unsigned realSizeLastDim, BaseAccessor acc = BaseAccessor()):
            base_(static_cast<ParamType>(data)), acc_(std::move(acc))
        {
            policies::GetExtents<Base> extents(base_);
            for(unsigned i=0; i<numDims; ++i)
                extents_[i] = extents[i];
            if((FFT_Def::kind == FFT_Kind::Complex2Real && isInput) ||
                    (FFT_Def::kind == FFT_Kind::Real2Complex && !isInput) ||
                    FFT_Def::kind == FFT_Kind::Complex2Complex)
            {
                if(extents_[numDims - 1] != realSizeLastDim)
                    throw std::runtime_error("Invalid size given");
            }else if((FFT_Def::kind == FFT_Kind::Complex2Real && !isInput) ||
                    (FFT_Def::kind == FFT_Kind::Real2Complex && isInput))
            {
                if(extents_[numDims - 1] != realSizeLastDim / 2 + 1)
                    throw std::runtime_error("Invalid size given");
            }
            if(FFT_Def::isInplace)
            {
                realExtents_ = extents_;
                realExtents_[numDims - 1] = realSizeLastDim;
            }

            memory_.init(extents_);
            if(memory_.checkPtr(base_, acc_, FFT_Def::isInplace && !isComplex))
                memFallback_ = nullptr;
            else if(FFT_Def::isInplace)
                throw std::runtime_error("Cannot use given memory as the strides/indexing is wrong!");
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
        size_t
        getNumElements() const
        {
            return std::accumulate(extents_.cbegin(), extents_.cend(), 1u, std::multiplies<size_t>());
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
            Parent::preProcess();
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
            Parent::postProcess();
        }
    };

    template< class T_FFT_Def, typename T_Base, class T_HasInstance = std::false_type, typename T_BaseAccessor = traits::IdentityAccessor_t<T_Base> >
    using FFT_InputDataWrapper = FFT_DataWrapper< T_FFT_Def, std::true_type, T_Base, T_HasInstance, T_BaseAccessor >;

    template< class T_FFT_Def, typename T_Base, class T_HasInstance = std::false_type, typename T_BaseAccessor = traits::IdentityAccessor_t<T_Base> >
    using FFT_OutputDataWrapper = FFT_DataWrapper< T_FFT_Def, std::false_type, T_Base, T_HasInstance, T_BaseAccessor >;

}  // namespace foobar
