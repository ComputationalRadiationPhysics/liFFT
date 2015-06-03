#pragma once

#include "foobar/traits/all.hpp"
#include "foobar/policies/all.hpp"

template< typename T = double >
class Volume{
    T* data_;
    bool isOwned_;
    const size_t xDim_, yDim_, zDim_;
public:
    using value_type = T;

    Volume(size_t xDim, size_t yDim, size_t zDim): xDim_(xDim), yDim_(yDim), zDim_(zDim){
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
    T* data(){
        return data_;
    }
    T& operator()(size_t x, size_t y=0, size_t z=0){
        return data_[(z*yDim_ + y)*xDim_ + x];
    }
    const T& operator()(size_t x, size_t y=0, size_t z=0) const{
        return data_[(z*yDim_ + y)*xDim_ + x];
    }

    size_t xDim() const{ return xDim_; }
    size_t yDim() const{ return yDim_; }
    size_t zDim() const{ return zDim_; }
};

namespace foobar {
    namespace traits {

        template<typename T>
        struct MemoryType< Volume<T> >
        {
            using type = Volume<T>;
        };

        template<typename T>
        struct IntegralTypeImpl< Volume<T> >: IntegralType< T >{}; // or define "type = T" in Volume itself

        //template<typename T>
        //struct NumDims< Volume<T> >: std::integral_constant< unsigned, 3 >{};

        template<typename T>
        struct IsComplex< Volume<T> >: IsComplex<T>{};

        template<typename T>
        struct IsStrided< Volume<T> >: std::false_type{};

        template<typename T>
        struct IsAoS< Volume<T> >: IsComplex<T>{};

    }  // namespace traits

    namespace policies {

        template<typename T>
        struct GetRawPtr< Volume<T> >: boost::noncopyable
        {
            using type = Volume<T>;
            using IntegralType = typename traits::IntegralType<T>::type;

            GetRawPtr(const type& data): data_( reinterpret_cast<IntegralType*>(const_cast<type&>(data).data()) ){}

            IntegralType*
            operator()(){
                return data_;
            }

        private:
            IntegralType* data_;
        };

        template<typename T>
        struct GetExtents< Volume<T> >: boost::noncopyable
        {
            using Data = Volume<T>;

            GetExtents(const Data& data): data_(data){}

            unsigned operator[](unsigned dimIdx)
            {
                switch(dimIdx){
                case 0:
                    return data_.xDim();
                case 1:
                    return data_.yDim();
                case 2:
                    return data_.zDim();
                }
                throw std::logic_error("Invalid dimension");
            }
        protected:
            const Data& data_;;
        };

        template<typename T, unsigned T_numDims >
        struct GetExtentsRawPtr< Volume<T>, T_numDims >: GetExtentsRawPtrImpl< Volume<T>, true, T_numDims >{
            using Parent = GetExtentsRawPtrImpl< Volume<T>, true, T_numDims >;

            GetExtentsRawPtr(const Volume<T>& data): Parent(data){}
        };

    }  // namespace policies
}  // namespace foobar

