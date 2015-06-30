#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/policies/Copy.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/mem/DataContainer.hpp"

struct GeneratorAccessor
{
    template< class T_Idx, class T_Data >
    std::result_of_t<std::remove_pointer_t<T_Data>(size_t, size_t, size_t)>
    operator()(T_Idx&& idx, const T_Data& data)
    {
        static constexpr unsigned numDims = foobar::traits::NumDims<T_Idx>::value;
        size_t z = (numDims < 3) ? 0 : idx[numDims - 3];
        size_t y = (numDims < 2) ? 0 : idx[numDims - 2];
        size_t x = idx[numDims - 1];
        return (*data)(x, y, z);
    }
};

template< typename T, class Generator, class T_Accessor = foobar::traits::DefaultAccessor_t<T> >
void generateData(T& data, const Generator& generator, const T_Accessor& acc = T_Accessor()){
    static constexpr unsigned numDims = foobar::traits::NumDims<T>::value;
    foobar::mem::DataContainer< numDims, const Generator*, GeneratorAccessor, false > genContainer;
    genContainer.data = &generator;
    foobar::policies::GetExtents<T> extents(data);
    for(unsigned i=0; i<numDims; i++)
        genContainer.extents[i] = extents[i];
    foobar::policies::makeCopy(typename decltype(genContainer)::Accessor(), acc)(genContainer, data);
}

template<typename T>
struct Spalt{
    const int size_;
    Spalt(int size):size_(size){}

    T
    operator()(size_t x, size_t y, size_t z) const{
        return (abs(x-500)<=size_) ? 1 : 0;
    }
};

template<typename T>
struct Rect{
    const int sizeX_, sizeY_;
    Rect(int sizeX, int sizeY):sizeX_(sizeX), sizeY_(sizeY){}

    T
    operator()(size_t x, size_t y, size_t z) const{
        return (abs(x-500)<=sizeX_ && abs(y-500)<=sizeY_) ? 1 : 0;
    }
};

template<typename T>
struct Circle{
    const int size_;
    Circle(int size):size_(size){}

    T
    operator()(size_t x, size_t y, size_t z) const{
        return (pow(abs(x-500), 2)+pow(abs(y-500), 2)<=size_*size_) ? 1 : 0;
    }
};

template<typename T>
struct Nullify{
    T
    operator()(size_t x, size_t y, size_t z) const{
        return 0;
    }
};
