#pragma once

template< typename T, class Generator>
void generateData(Volume<T>& data, const Generator& generator){
    for(size_t z=0; z<data.zDim(); ++z){
        for(size_t y=0; y<data.yDim(); ++y){
            for(size_t x=0; x<data.xDim(); ++x)
                data(x,y,z) = generator(x,y,z);
        }
    }
}

template<typename T>
struct Spalt{
    const int size_;
    Spalt(int size):size_(size){}

    T
    operator()(size_t x, size_t y, size_t z) const{
        return (abs(x-500)<size_) ? 1 : 0;
    }
};

template<typename T>
struct Rect{
    const int sizeX_, sizeY_;
    Rect(int sizeX, int sizeY):sizeX_(sizeX), sizeY_(sizeY){}

    T
    operator()(size_t x, size_t y, size_t z) const{
        return (abs(x-500)<sizeX_ && abs(y-500)<sizeY_) ? 1 : 0;
    }
};

template<typename T>
struct Circle{
    const int size_;
    Circle(int size):size_(size){}

    T
    operator()(size_t x, size_t y, size_t z) const{
        return (pow(abs(x-500), 2)+pow(abs(y-500), 2)<size_*size_) ? 1 : 0;
    }
};

template<typename T>
struct Nullify{
    T
    operator()(size_t x, size_t y, size_t z) const{
        return 0;
    }
};
