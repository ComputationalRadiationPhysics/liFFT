#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IdentityAccessor.hpp"
#include "foobar/policies/Copy.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/mem/DataContainer.hpp"

namespace foobar {

    struct GeneratorAccessor
    {
        template< class T_Idx, class T_Data >
        std::result_of_t<std::remove_pointer_t<T_Data>(T_Idx)>
        operator()(T_Idx&& idx, const T_Data& data)
        {
            return (*data)(std::forward<T_Idx>(idx));
        }
    };

    template< typename T, class Generator, class T_Accessor = foobar::traits::IdentityAccessor_t<T> >
    void generateData(T& data, const Generator& generator, const T_Accessor& acc = T_Accessor()){
        static constexpr unsigned numDims = foobar::traits::NumDims<T>::value;
        types::Vec<numDims> extents;
        policies::GetExtents<T> extentsData(data);
        for(unsigned i=0; i<numDims; i++)
            extents[i] = extentsData[i];
        mem::DataContainer< numDims, const Generator*, GeneratorAccessor, false > genContainer(&generator, extents);
        policies::makeCopy(foobar::traits::IdentityAccessor_t< decltype(genContainer) >(), acc)(genContainer, data);
    }

    namespace generators {

        template<typename T>
        struct Spalt{
            const size_t size_, middle_;
            Spalt(size_t size, size_t middle):size_(size), middle_(middle){}

            template< class T_Idx >
            T
            operator()(T_Idx&& idx) const{
                static constexpr unsigned numDims = traits::NumDims<T_Idx>::value;

                return (std::abs(idx[numDims - 1] - middle_) <= size_) ? 1 : 0;
            }
        };

        template<typename T>
        struct Cosinus{
            const size_t middle_;
            const T factor_;
            Cosinus(size_t period, size_t middle):middle_(middle), factor_(2 * M_PI / period){}

            template< class T_Idx >
            T
            operator()(T_Idx&& idx) const{
                static constexpr unsigned numDims = traits::NumDims<T_Idx>::value;
                static_assert(numDims >= 2, "Only >=2D data supported");

                auto dist = std::sqrt(
                        std::pow(std::abs(idx[numDims - 1] - middle_), 2)+
                        std::pow(std::abs(idx[numDims - 2] - middle_), 2));
                return std::cos( factor_ * dist );
            }
        };

        template<typename T>
        struct Rect{
            const size_t sizeX_, sizeY_, middleX_, middleY_;
            Rect(size_t sizeX, size_t middleX): Rect(sizeX, middleX, sizeX, middleX){}
            Rect(size_t sizeX, size_t middleX, size_t sizeY, size_t middleY):sizeX_(sizeX), sizeY_(sizeY), middleX_(middleX), middleY_(middleY){}

            template< class T_Idx >
            T
            operator()(T_Idx&& idx) const{
                static constexpr unsigned numDims = traits::NumDims<T_Idx>::value;
                static_assert(numDims >= 2, "Only >=2D data supported");

                return (std::abs(idx[numDims - 1] - middleX_) <= sizeX_ &&
                        std::abs(idx[numDims - 2] - middleY_) <= sizeY_) ? 1 : 0;
            }
        };

        template<typename T>
        struct Circle{
            const size_t size_, middle_;
            Circle(size_t size, size_t middle):size_(size), middle_(middle){}

            template< class T_Idx >
            T
            operator()(T_Idx&& idx) const{
                static constexpr unsigned numDims = traits::NumDims<T_Idx>::value;
                static_assert(numDims >= 2, "Only >=2D data supported");

                return (std::pow(std::abs(idx[numDims - 1] - middle_), 2) +
                        std::pow(std::abs(idx[numDims - 2] - middle_), 2) <= size_*size_) ? 1 : 0;
            }
        };

        template<typename T>
        struct SetToConst{
            const T val_;
            SetToConst(T val): val_(val){}

            template< class T_Idx >
            T
            operator()(T_Idx&&) const{
                return val_;
            }
        };

    }  // namespace generators

}  // namespace foobar
