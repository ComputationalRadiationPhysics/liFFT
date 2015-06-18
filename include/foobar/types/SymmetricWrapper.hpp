#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/GetExtents.hpp"

namespace foobar {
    namespace types {

        /**
         * Wrapper class to access types that are symmetric in the last dimension
         */
        template< class T_Base, class T_Accessor >
        struct SymmetricWrapper
        {
            using Base = T_Base;
            using Accessor = T_Accessor;

            static constexpr unsigned numDims = traits::NumDims<Base>::value;

            SymmetricWrapper(Base& base, unsigned realSize): base_(base), realSize_(realSize){}

            template< typename T_Index >
            auto
            operator()(const T_Index& idx) const
            -> std::result_of_t< Accessor(const T_Index&, const Base&) >
            {
                static constexpr unsigned lastDim = numDims - 1;
                // If this instance is const, the base type (and therefore the returned type)
                // must also be const, but base_ is a reference and therefore not const yet
                const Base& cBase = base_;
                policies::GetExtents<Base> extents(cBase);
                if(idx[lastDim] >= extents[lastDim]){
                    T_Index newIdx(idx);
                    newIdx[lastDim] = realSize_ - idx[lastDim];
                    return acc_(newIdx, cBase);
                }else
                    return acc_(idx, cBase);
            }

            template< typename T_Index >
            auto
            operator()(const T_Index& idx)
            -> std::result_of_t< Accessor(const T_Index&, Base&) >
            {
                static constexpr unsigned lastDim = numDims - 1;
                policies::GetExtents<Base> extents(base_);
                if(idx[lastDim] >= extents[lastDim]){
                    T_Index newIdx(idx);
                    newIdx[lastDim] = realSize_ - idx[lastDim];
                    return acc_(newIdx, base_);
                }else
                    return acc_(idx, base_);
            }
        private:
            Base& base_;
            Accessor acc_;
            unsigned realSize_;
            friend struct policies::GetExtents<SymmetricWrapper>;
        };

    }  // namespace types

    namespace policies {

        template< class T_Base, class T_Accessor >
        struct GetExtents< types::SymmetricWrapper< T_Base, T_Accessor> >: private boost::noncopyable
        {
            using Data = types::SymmetricWrapper< T_Base, T_Accessor>;
            using Extents = GetExtents<T_Base>;
            static constexpr unsigned numDims = traits::NumDims<Data>::value;

            GetExtents(const Data& data): data_(data){}

            unsigned operator[](unsigned dimIdx) const
            {
                if(dimIdx == numDims-1)
                    return data_.realSize_;
                else
                    return GetExtents<T_Base>(data_.base_)[dimIdx];
            }
        protected:
            const Data& data_;
        };

    }  // namespace policies
}  // namespace foobar
