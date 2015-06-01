#pragma once

namespace LibFoo
{
    void calculateR1D(double* in, double* out, unsigned numEl){
        std::cout << "Real 1D output of " << numEl << " elements" << std::endl;
    }

    void calculateRND(double* in, double* out, unsigned numDims, unsigned* numEl){
        std::cout << "Real " << numDims << "D output of (";
        for(unsigned i=0; i<numDims; ++i)
            std::cout << numEl[i] << ", ";
        std::cout << ") elements" << std::endl;
    }

    void calculateC1D(double* in, double* out, unsigned numEl){
        std::cout << "Complex 1D output of " << numEl << " elements" << std::endl;
    }

    void calculateCND(double* in, double* out, unsigned numDims, unsigned* numEl){
        std::cout << "Complex " << numDims << "D output of (";
        for(unsigned i=0; i<numDims; ++i)
            std::cout << numEl[i] << ", ";
        std::cout << ") elements" << std::endl;
    }

    void calculateC1D(double* inReal, double* inImag, double* out, unsigned numEl){
        std::cout << "Complex(SoA) 1D output of " << numEl << " elements" << std::endl;
    }

    void calculateCND(double* inReal, double* inImag, double* out, unsigned numDims, unsigned* numEl){
        std::cout << "Complex(SoA) " << numDims << "D output of (";
        for(unsigned i=0; i<numDims; ++i)
            std::cout << numEl[i] << ", ";
        std::cout << ") elements" << std::endl;
    }
} // namespace LibFoo
