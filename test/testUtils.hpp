#pragma once

#include "testDefines.hpp"
#include "foobar/traits/IdentityAccessor.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/policies/Loop.hpp"
#include <cmath>
#include <iostream>

namespace foobarTest {

    /**
     * Initializes the test environment and prepares the base line FFTs to compare against
     */
    void init();
    /**
     * Frees all resources used in the test environment
     */
    void finalize();
    /**
     * Executes the base tests and outputs PDFs visualizing the results
     */
    void visualizeBase();

    enum class BaseInstance{
        InR2C, OutR2C,
        InC2C, OutC2C
    };

    /**
     * Outputs one of the base data containers to a file (pdf+txt)
     *
     * @param inst Which one to output
     * @param filePath Output path
     */
    void visualizeOutput(BaseInstance inst, const std::string& filePath);

    /**
     * Executes the FFT on the base Input for R2C
     */
    void execBaseR2C();
    /**
     * Executes the FFT on the base Input for C2C
     */
    void execBaseC2C();

    /**
     * Maximum error detected during a compare run
     */
    struct CmpError{
        double maxAbsDiff = 0;
        double maxRelDiff = 0;

        CmpError(): maxAbsDiff(0), maxRelDiff(0)
        {}
        CmpError(double allowedAbsDiff, double allowedRelDiff): maxAbsDiff(allowedAbsDiff), maxRelDiff(allowedRelDiff)
        {}

        friend inline std::ostream& operator<<(std::ostream& stream, const CmpError& e){
            stream << "Max AbsDiff = " << e.maxAbsDiff  << " Max RelDiff = " << e.maxRelDiff;
            return stream;
        }
    };


    /**
     * "Functor" used for comparing multidimensional containers of Real or Complex data
     */
    struct CompareFunc
    {
        bool ok = true;
        CmpError e;
        CmpError allowed_;

        CompareFunc(CmpError allowed): allowed_(allowed){}

        template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
        void handleLoopPre(T_Args&&...){}
        template< unsigned T_curDim, unsigned T_endDim, class... T_Args>
        void handleLoopPost(T_Args&&...){}

        template<class T, class U>
        std::enable_if_t< foobar::traits::IsComplex<T>::value, bool >
        compare(const T& expected, const U& is)
        {
            using Precision = foobar::traits::IntegralType_t<T>;
            using Complex = foobar::types::Complex<Precision>;
            static_assert(foobar::traits::IsBinaryCompatible<T, Complex>::value, "Cannot convert expected");
            static_assert(foobar::traits::IsBinaryCompatible<U, Complex>::value, "Cannot convert is");
            const Complex& expected_ = reinterpret_cast<const Complex&>(expected);
            const Complex& is_ = reinterpret_cast<const Complex&>(is);
            return compare(expected_.real, is_.real) && compare(expected_.imag, is_.imag);
        }

        template<class T, class U>
        std::enable_if_t< !foobar::traits::IsComplex<T>::value, bool >
        compare(const T& expected, const U& is)
        {
            if(expected == is)
                return true;
            auto absDiff = std::abs(expected-is);
            if(absDiff <= allowed_.maxAbsDiff)
                return true;
            auto relDiff = std::abs(absDiff / expected);
            if(relDiff <= allowed_.maxRelDiff)
                return true;
            if(absDiff > e.maxAbsDiff)
                e.maxAbsDiff = absDiff;
            if(relDiff > e.maxRelDiff)
                e.maxRelDiff = relDiff;
            return false;
        }

        template<
            unsigned T_curDim,
            class T_Index,
            class T_Src,
            class T_SrcAccessor,
            class T_Dst,
            class T_DstAccessor
            >
        void
        handleInnerLoop(const T_Index& idx, const T_Src& expected, T_SrcAccessor&& accExp, T_Dst& is, T_DstAccessor&& accIs)
        {
            if(!compare(accExp(idx, expected), accIs(idx, is)))
                ok = false;
        }
    };

    /**
     * Compares 2 multidimensional containers
     *
     * @param expected   Container with expected values
     * @param is         Container with actual values
     * @param allowedErr Maximum allowed error
     * @param expAcc     Accessor for expected container [IdentityAccessor used]
     * @param isAcc      Accessor for actual container [IdentityAccessor used]
     * @return Pair: 1: bool OK, 2: Maximum errors detected
     */
    template< class T, class U, class T_AccessorT = foobar::traits::IdentityAccessor_t<T>, class T_AccessorU = foobar::traits::IdentityAccessor_t<U> >
    std::pair< bool, CmpError >
    compare(const T& expected, const U& is, CmpError allowedErr = CmpError(1e-4, 5e-5), const T_AccessorT& expAcc = T_AccessorT(), const T_AccessorU& isAcc = T_AccessorU())
    {
        CompareFunc result(allowedErr);
        foobar::policies::loop(expected, result, expAcc, is, isAcc);
        return std::make_pair(result.ok, result.e);
    }

    /**
     * Checks if the results match and prints a message about the result
     *
     * @param baseRes   Result from base execution (assumed valid)
     * @param res       Data to compare against
     * @param testDescr String the identifies the test
     * @param maxErr    Maximum allowed error
     * @return True on success
     */
    template< class T_BaseResult, class T_Result >
    bool checkResult(const T_BaseResult& baseRes, const T_Result& res, const std::string& testDescr, CmpError maxErr = CmpError(1e-4, 5e-5))
    {
        auto cmpRes = compare(baseRes, res, maxErr);
        if(!cmpRes.first)
            std::cerr << "Error for " << testDescr << ": " << cmpRes.second << std::endl;
        else
            std::cout << testDescr << " passed" << std::endl;
        return cmpRes.first;
    }

}  // namespace foobarTest
