#pragma once

#include "policyExample.hpp"
#include <chrono>
#include <vector>
#include "foobar/mem/DataContainer.hpp"
#include "foobar/mem/RealValues.hpp"
#include "foobar/mem/ComplexAoSValues.hpp"
#include "foobar/mem/ComplexSoAValues.hpp"


using millisecs = std::chrono::duration<unsigned long long, std::milli>;
using std::chrono::duration_cast;
using MyClock = std::chrono::high_resolution_clock;

constexpr unsigned NumVals = 1e7;

template< class T >
bool checkResult(const T& result, unsigned expected){
    bool ok = true;
    for(auto val:result){
        if(val != 4*4+3*3){
            ok = false;
            break;
        }
    }

    return ok;
}

void test(){
    using namespace foobar::types;
    using namespace foobar::mem;
    using foobar::calcIntensity;
    using foobar::calcIntensity2;

    DataContainer< 1, RealValues<double> > simpleRealData;
    DataContainer< 1, ComplexAoSValues<double> > complexAoS;
    DataContainer< 1, ComplexSoAValues<double> > complexSoA;

    simpleRealData.extents = {NumVals};
    simpleRealData.data = new Real<double>[NumVals];
    for(unsigned i=0; i<NumVals; i++)
        simpleRealData.data[i] = 5;

    complexAoS.extents = {NumVals};
    complexAoS.data = new Complex<double>[NumVals];
    for(unsigned i=0; i<NumVals; i++){
        complexAoS.data[i].real = 4;
        complexAoS.data[i].imag = 3;
    }

    complexSoA.extents = {NumVals};
    complexSoA.data.getRealData() = new Real<double>[NumVals];
    complexSoA.data.getImagData() = new Real<double>[NumVals];
    for(unsigned i=0; i<NumVals; i++){
        complexSoA.data.getRealData()[i] = 4;
        complexSoA.data.getImagData()[i] = 3;
    }

    std::vector<double> result(NumVals);

    MyClock timer;
    auto t1 = timer.now();
    calcIntensity(simpleRealData, result.data());
    if(!checkResult(result, 5*5))
        std::cout << "Check failed :(";
    calcIntensity(complexAoS, result.data());
    if(!checkResult(result, 4*4+3*3))
        std::cout << "Check failed :(";
    calcIntensity(complexSoA, result.data());
    if(!checkResult(result, 4*4+3*3))
        std::cout << "Check failed :(";
    auto t2 = timer.now();
    std::cout << "Time used for 1: " << duration_cast<millisecs>(t2-t1).count() << "ms\n";

    t1 = timer.now();
    calcIntensity2(simpleRealData, result.data());
    if(!checkResult(result, 5*5))
        std::cout << "Check failed :(";
    calcIntensity2(complexAoS, result.data());
    if(!checkResult(result, 4*4+3*3))
        std::cout << "Check failed :(";
    calcIntensity2(complexSoA, result.data());
    if(!checkResult(result, 4*4+3*3))
        std::cout << "Check failed :(";
    t2 = timer.now();
    std::cout << "Time used for 2: " << duration_cast<millisecs>(t2-t1).count() << "ms\n";
}
