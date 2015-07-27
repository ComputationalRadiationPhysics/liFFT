#pragma once

#include "foobar/accessors/ArrayAccessor.hpp"
#include "foobar/FFT_DataWrapper.hpp"
#include "foobar/c++14_types.hpp"
#include "foobar/traits/IdentityAccessor.hpp"
#include "foobar/mem/RealValues.hpp"
#include "foobar/mem/ComplexAoSValues.hpp"
#include "foobar/mem/ComplexSoAValues.hpp"
#include "foobar/mem/DataContainer.hpp"

namespace foobar {

    /**
     * Class used to access the output of an inplace FFT
     * \tparam T_FFT_Input InputDataWrapper
     */
    template< class T_FFT_Input >
    class FFT_InplaceOutput: public detail::IInplaceOutput
    {
        using Input = T_FFT_Input;
        using FFT_Def = typename Input::FFT_Def;
        using InputAccessor = typename Input::IdentityAccessor;

    public:
        static constexpr unsigned numDims = Input::numDims;
        static constexpr bool isComplex   = FFT_Def::isComplexOutput;
        static constexpr bool isAoS       = Input::isAoS;
        static constexpr bool isStrided   = Input::isStrided;

        using IdxType = typename Input::IdxType;
        using Extents = typename Input::Extents;
        using PrecisionType = typename Input::PrecisionType;

    private:
        using Values = std::conditional_t<
                            isComplex,
                            std::conditional_t<
                                isAoS,
                                mem::ComplexAoSValues<PrecisionType, false>,
                                mem::ComplexSoAValues<PrecisionType, false>
                            >,
                            mem::RealValues<PrecisionType, false>
                        >;
        using Data = mem::DataContainer<numDims, Values, traits::IdentityAccessor_t<Values>, true, isStrided>;

    public:
        using IdentityAccessor = accessors::ArrayAccessor<true>;

        FFT_InplaceOutput(Input& input): input_(input)
        {
            updateData();
        }

        std::result_of_t< Data(const IdxType&) >
        operator()(const IdxType& idx)
        {
            return data_(idx);
        }

        std::result_of_t< const Data(const IdxType&) >
        operator()(const IdxType& idx) const
        {
            return data_(idx);
        }

        const Extents&
        getExtents() const
        {
            return data_.getExtents();
        }

        void preProcess(){}
        void postProcess(){
            updateData();
        }
    private:
        Input& input_;
        Data data_;

        void updateData(){
            size_t numElements =  std::accumulate(input_.fullExtents_.cbegin(), input_.fullExtents_.cend(), 1u, std::multiplies<size_t>());
            if(traits::getMemSize(input_) < numElements * sizeof(typename Values::Value))
                throw std::runtime_error("Number of elements is wrong or not enough memory allocated");
            data_.setData(input_.fullExtents_, Values(input_.getDataPtr(), numElements));
        }
    };

    namespace detail {


    }  // namespace detail

}  // namespace foobar
