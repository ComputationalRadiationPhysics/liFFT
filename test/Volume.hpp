#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/policies/GetExtents.hpp"

template< typename T = double >
class Volume{
    T* data_;
    bool isOwned_;
    const size_t xDim_, yDim_, zDim_;
public:
    using value_type = T;
    using Ref = T&;
    using ConstRef = const T&;

    Volume(size_t xDim, size_t yDim = 1, size_t zDim = 1): xDim_(xDim), yDim_(yDim), zDim_(zDim){
        data_ = static_cast<T*>(fftw_malloc(xDim*yDim*zDim*sizeof(T))); //new T[xDim*yDim*zDim_];
        isOwned_ = true;
    }
    Volume(size_t xDim, size_t yDim, size_t zDim, T* data): xDim_(xDim), yDim_(yDim), zDim_(zDim){
        data_ = data;
        isOwned_ = false;
    }
    ~Volume(){
        if(isOwned_)
            fftw_free(data_);//delete[] data_;
    }
    T*
    data(){
        return data_;
    }
    Ref
    operator()(size_t x, size_t y=0, size_t z=0){
        return data_[(z*yDim_ + y)*xDim_ + x];
    }
    ConstRef
    operator()(size_t x, size_t y=0, size_t z=0) const{
        return data_[(z*yDim_ + y)*xDim_ + x];
    }

    size_t xDim() const{ return xDim_; }
    size_t yDim() const{ return yDim_; }
    size_t zDim() const{ return zDim_; }
};

namespace foobar {
    namespace traits {

        template<typename T>
        struct IntegralTypeImpl< Volume<T> >: IntegralType< T >{}; // or define "type = T" in Volume itself

        template<typename T>
        struct NumDims< Volume<T> >: std::integral_constant< unsigned, 3 >{};

        template<typename T>
        struct IsComplex< Volume<T> >: IsComplex<T>{};

        template<typename T>
        struct IsStrided< Volume<T> >: std::false_type{};

        template<typename T>
        struct IsAoS< Volume<T> >: std::true_type{};

    }  // namespace traits

    namespace policies {

        template< class T_Data >
        struct GetVolumeExtents: boost::noncopyable
        {
            using Data = T_Data;

            GetVolumeExtents(const Data& data): data_(data){}

            unsigned operator[](unsigned dimIdx) const
            {
                switch(dimIdx){
                case 0:
                    return data_.zDim();
                case 1:
                    return data_.yDim();
                case 2:
                    return data_.xDim();
                }
                throw std::logic_error("Invalid dimension");
            }
        protected:
            const Data& data_;
        };

        template<typename T>
        struct GetExtents< Volume<T> >: GetVolumeExtents< Volume<T> >
        {
            using Parent = GetVolumeExtents< Volume<T> >;
            using Parent::Parent;
        };

    }  // namespace policies
}  // namespace foobar

