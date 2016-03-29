#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IdentityAccessor.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/accessors/ArrayAccessor.hpp"
#include "foobar/types/Complex.hpp"
#include "foobar/c++14_types.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/GetMemSize.hpp"

namespace foobar {
    namespace types {

        /**
         * Wrapper class to access types that are symmetric in the last dimension
         */
        template< class T_Base, class T_BaseAccessor = traits::IdentityAccessor_t<T_Base> >
        struct SymmetricWrapper
        {
            using Base = T_Base;
            using BaseAccessor = T_BaseAccessor;

            using IdentityAccessor = accessors::ArrayAccessor<true>;

            static constexpr unsigned numDims = traits::NumDims<Base>::value;

            SymmetricWrapper(Base& base, unsigned realSize): m_base(base), m_realSize(realSize){}

            template< typename T >
            static Complex<T>
            makeConjugate(const Complex<T>& val)
            {
                return Complex<T>(val.real, -val.imag);
            }

            template< typename T >
            static T
            makeConjugate(const T& val)
            {
                return val;
            }

            template< typename T_Index >
            auto
            operator()(const T_Index& idx) const
            -> std::remove_reference_t< std::result_of_t< BaseAccessor(const T_Index&, const Base&) > >
            {
                static constexpr unsigned lastDim = numDims - 1;
                // If this instance is const, the base type (and therefore the returned type)
                // must also be const, but m_base is a reference and therefore not const yet
                const Base& cBase = m_base;
                policies::GetExtents<Base> extents(cBase);
                if(idx[lastDim] >= extents[lastDim]){
                    // We are in the "virtual" part, that does not exist in memory
                    // --> Use symmetry as conj(F(x,y,z)) = F(-x,-y,-z) = F(nx-x, ny-y, nz-z)
                    // And be careful with the 0 element as F(nx, ny, nz) does not exist in memory, but is (periodicity) F(0, 0, 0)
                    T_Index newIdx;
                    newIdx[lastDim] = m_realSize - idx[lastDim];
                    for(unsigned i=0; i<lastDim; i++)
                        newIdx[i] = idx[i] == 0 ? 0 : extents[i] - idx[i];
                    return makeConjugate(m_acc(newIdx, cBase));
                }else
                    return m_acc(idx, cBase);
            }

            template<typename T>
            struct DisableForComplex
            {
                using Plain = std::remove_cv_t<T>;
                static constexpr bool isComplex = traits::IsComplex<Plain>::value;
                using type = std::enable_if< !isComplex, T >;
            };

            // Only enabled for non-complex types as references are not possible due to "virtual memory handling" with conjugate complex values
            template< typename T_Index >
            auto
            operator()(const T_Index& idx)
            -> typename DisableForComplex< std::result_of_t< BaseAccessor(const T_Index&, Base&) > >::type::type
            {
                static constexpr unsigned lastDim = numDims - 1;
                policies::GetExtents<Base> extents(m_base);
                if(idx[lastDim] >= extents[lastDim]){
                    T_Index newIdx(idx);
                    newIdx[lastDim] = m_realSize - idx[lastDim];
                    return m_acc(newIdx, m_base);
                }else
                    return m_acc(idx, m_base);
            }

            size_t
            getMemSize() const
            {
                return traits::getMemSize(m_base);
            }
        private:
            Base& m_base;
            BaseAccessor m_acc;
            unsigned m_realSize;
            friend struct policies::GetExtentsImpl<SymmetricWrapper>;
        };

        template< class T_Base, class T_BaseAccessor = traits::IdentityAccessor_t<T_Base> >
        SymmetricWrapper< T_Base, T_BaseAccessor >
        makeSymmetricWrapper(T_Base& base, unsigned realSize)
        {
            return SymmetricWrapper< T_Base, T_BaseAccessor >(base, realSize);
        }

    }  // namespace types

    namespace policies {

        template< class T_Base, class T_Accessor >
        struct GetExtentsImpl< types::SymmetricWrapper< T_Base, T_Accessor> >: private boost::noncopyable
        {
            using Data = types::SymmetricWrapper< T_Base, T_Accessor>;
            using Extents = GetExtents<T_Base>;
            static constexpr unsigned numDims = traits::NumDims<Data>::value;

            GetExtentsImpl(const Data& data): m_data(data){}

            unsigned operator[](unsigned dimIdx) const
            {
                if(dimIdx == numDims-1)
                    return m_data.m_realSize;
                else
                    return GetExtents<T_Base>(m_data.m_base)[dimIdx];
            }
        protected:
            const Data& m_data;
        };

    }  // namespace policies
}  // namespace foobar
