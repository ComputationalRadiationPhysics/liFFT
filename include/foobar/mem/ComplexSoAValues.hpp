#pragma once

#include "foobar/mem/RealValues.hpp"
#include "foobar/types/Complex.hpp"
#include "foobar/policies/ArrayAccessor.hpp"

namespace foobar {
    namespace mem {

        template< typename T, bool T_ownsPointer = true >
        class ComplexSoAValues
        {
        public:
            using type = T;
            static constexpr bool isComplex = true;
            static constexpr bool isAoS = false;
            static constexpr bool ownsPointer = T_ownsPointer;
            using Data = RealValues<T, ownsPointer>;
            using Ptr = typename Data::Ptr;
            using Value = types::Complex<T>;
            using Ref = types::ComplexRef<T>;
            using ConstRef = const types::ComplexRef<T, true>;
            using Accessor = policies::ArrayAccessor<>;

            ComplexSoAValues(){}
            ComplexSoAValues(Ptr realData, Ptr imagData): real_(realData), imag_(imagData){}

            void
            operator=(std::pair<Ptr, Ptr> data)
            {
                real_ = data.first;
                imag_ = data.second;
            }

            void
            allocData(size_t numElements)
            {
                real_.allocData(numElements);
                imag_.allocData(numElements);
            }

            void
            freeData()
            {
                real_.freeData();
                imag_.freeData();
            }

            void
            releaseData()
            {
                real_.releaseData();
                imag_.releaseData();
            }

            std::pair<Ptr, Ptr>
            getData() const
            {
                return std::make_pair(real_.getData(), imag_.getData());
            }

            ConstRef
            operator[](size_t idx) const
            {
                return ConstRef(real_[idx], imag_[idx]);
            }

            Ref
            operator[](size_t idx)
            {
                return Ref(real_[idx], imag_[idx]);
            }

            Data&
            getRealData()
            {
                return real_;
            }

            Data&
            getImagData()
            {
                return imag_;
            }

        private:
            Data real_, imag_;
        };

    }  // namespace types

}  // namespace foobar
