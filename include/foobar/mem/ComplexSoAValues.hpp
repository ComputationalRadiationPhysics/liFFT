#pragma once

#include "foobar/mem/RealValues.hpp"
#include "foobar/types/Complex.hpp"
#include "foobar/accessors/ArrayAccessor.hpp"

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
            using IdentityAccessor = accessors::ArrayAccessor<>;

            ComplexSoAValues(){}
            ComplexSoAValues(Ptr realData, Ptr imagData, size_t numElements): m_real(realData, numElements), m_imag(imagData, numElements){}

            void
            reset(std::pair<Ptr, Ptr> data, size_t numElements)
            {
                m_real.reset(data.first, numElements);
                m_imag.reset(data.second, numElements);
            }

            void
            allocData(size_t numElements)
            {
                m_real.allocData(numElements);
                m_imag.allocData(numElements);
            }

            void
            freeData()
            {
                m_real.freeData();
                m_imag.freeData();
            }

            std::pair<Ptr, Ptr>
            releaseData()
            {
                return std::make_pair(m_real.releaseData(), m_imag.releaseData());
            }

            std::pair<Ptr, Ptr>
            getData() const
            {
                return std::make_pair(m_real.getData(), m_imag.getData());
            }

            ConstRef
            operator[](size_t idx) const
            {
                return ConstRef(m_real[idx], m_imag[idx]);
            }

            Ref
            operator[](size_t idx)
            {
                return Ref(m_real[idx], m_imag[idx]);
            }

            Data&
            getRealData()
            {
                return m_real;
            }

            Data&
            getImagData()
            {
                return m_imag;
            }

        private:
            Data m_real, m_imag;
        };

    }  // namespace mem

}  // namespace foobar
