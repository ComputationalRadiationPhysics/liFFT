#include <iostream>
#include <string>

#include "libTiff/image.hpp"
#include "libTiff/traitsAndPolicies.hpp"
#include "foobar/FFT.hpp"
#if defined(WITH_CUDA) and false
#   include "foobar/libraries/cuFFT/cuFFT.hpp"
    using FFT_LIB = foobar::libraries::cuFFT::CuFFT<>;
#else
#   include "foobar/libraries/fftw/FFTW.hpp"
    using FFT_LIB = foobar::libraries::fftw::FFTW<>;
#endif

#include "foobar/policies/Copy.hpp"
#include "foobar/accessors/TransformAccessor.hpp"
#include "foobar/accessors/TransposeAccessor.hpp"
#include "foobar/policies/CalcIntensityFunctor.hpp"
#include "foobar/generateData.hpp"
#include "foobar/types/SliceView.hpp"

#include <boost/program_options.hpp>
#include <cmath>

namespace po = boost::program_options;
using std::string;
using std::cout;
using std::cerr;

po::options_description desc("Generate 3D data, convert with FFT and output to tiff");

void showHelp()
{
    cout << desc << std::endl;
}

template<typename T>
inline T
gauss(T x, T xOffset, T scale, T delta)
{
    return scale * std::exp(-std::pow(x - xOffset, 2) / (T(2) * std::pow(delta, 2) ));
}

template<typename T>
inline T
raiseFunc(T x, T xOffset, T scale, T delta)
{
    return scale / (T(1) + std::exp(-(x - xOffset) * delta));
}

template<typename T>
inline T
sinFunc(T x, T xOffset, T scale, T freq)
{
    return scale * std::sin((x - xOffset) * freq);
}

template<typename T>
struct Values1_2P5um
{
    static constexpr T raise1Max = 0.65;
    static constexpr T raise1Delta = 0.01;
    static constexpr T raise1Offset = -100;
    static constexpr T raise2Max = 1;
    static constexpr T raise2Delta = 0.02;
    static constexpr T raise2Offset = 100;
    static constexpr T gaussMax = 2.4;
    static constexpr T gaussDelta = 150;
    static constexpr T gaussOffset = -350;
    static constexpr T scale = 0.55;
};

template<typename T>
struct Values1_1um
{
    static constexpr T raise1Max = 0.65;
    static constexpr T raise1Delta = 0.01;
    static constexpr T raise1Offset = -100;
    static constexpr T raise2Max = 1;
    static constexpr T raise2Delta = 0.046;
    static constexpr T raise2Offset = 40;
    static constexpr T gaussMax = 2.4;
    static constexpr T gaussDelta = 150;
    static constexpr T gaussOffset = -350;
    static constexpr T scale = 0.55;
};

template<typename T>
struct Values1_100nm
{
    static constexpr T raise1Max = 0.65;
    static constexpr T raise1Delta = 0.01;
    static constexpr T raise1Offset = -100;
    static constexpr T raise2Max = 1;
    static constexpr T raise2Delta = 0.381;
    static constexpr T raise2Offset = 4;
    static constexpr T gaussMax = 2.4;
    static constexpr T gaussDelta = 150;
    static constexpr T gaussOffset = -350;
    static constexpr T scale = 0.55;
};

template< typename T, template <typename U> class T_Values >
struct GenData1
{
    using Values = T_Values<T>;

    static constexpr T midPt = 1500;
    static constexpr T radius = 21200;
    static constexpr T x0 = 512u;
    static constexpr T y0 = x0;
    static constexpr T z0 = midPt - radius;

    T f(T x, T y, T z) const{
        int dx = x0 - x;
        int dy = y0 - y;
        int dz = z0 - z;
        unsigned r2 = dx*dx + dy*dy + dz*dz;
        T r = std::sqrt(r2) + z0;
        return Values::scale *
                raiseFunc<T>(r, midPt + Values::raise1Offset, Values::raise1Max, Values::raise1Delta) *
                (1 - raiseFunc<T>(r, midPt + Values::raise2Offset, Values::raise2Max, Values::raise2Delta)) *
                (1 - gauss<T>(r, midPt +Values::gaussOffset, Values::gaussMax, Values::gaussDelta));
    }

    template< class T_Idx >
    T
    operator()(T_Idx&& idx) const{
        static constexpr unsigned numDims = foobar::traits::NumDims<T_Idx>::value;
        static_assert(numDims == 3, "3D only");

        unsigned x = 512U;//idx[2];
        unsigned y = idx[1];
        unsigned z = idx[0];

        int dx = x0 - x;
        int dy = y0 - y;
        int dz = z0 - z;
        unsigned r2 = dx*dx + dy*dy + dz*dz;
        T r = std::sqrt(r2);
        T rand = T(std::rand()) / RAND_MAX;
        T rand2 = T(std::rand()) / RAND_MAX;
        T rand3 = T(std::rand()) / RAND_MAX;

        T res =  f(x, y, z);
        if(r < radius - 200)
            res += 0.07 *
                sinFunc<T>(x, 0, 1,1/(60. + rand)) *
                sinFunc<T>(y, 0, 1, 1/(60. + rand2)) *
                sinFunc<T>(z, midPt - 200, 1, 1/(30. + rand3));
        return res;
    }
};

template< class T >
void genData(T& data, unsigned dataSet)
{
    std::cout << "Generating Dataset " << dataSet << "..." << std::endl;
    switch(dataSet){
    case 1:
        foobar::generateData(data, GenData1<float, Values1_2P5um>());
        break;
    case 2:
        foobar::generateData(data, GenData1<float, Values1_1um>());
        break;
    case 3:
        foobar::generateData(data, GenData1<float, Values1_100nm>());
        break;
    default:
        throw std::logic_error("Wrong dataset");
    }
    std::cout << "Generating done" << std::endl;
}

void writeInput(const string& filePath, unsigned dataSet)
{
    libTiff::FloatImage<> img(filePath, 1024u, 2048u);
    foobar::mem::RealContainer<3, float> data(foobar::types::Vec3(2048u, 1024u, 1u));
    genData(data, dataSet);
    auto view = foobar::types::makeSliceView<2>(data, foobar::types::makeRange(foobar::types::Vec3(0u, 0u, 0u)));
    foobar::policies::copy(view, img);
    img.save();
}

void writeFFT(const string& filePath, unsigned dataSet)
{
    using FFT = foobar::FFT_3D_R2C_F<true>;
    auto input = FFT::createNewInput(foobar::types::Vec3(2048u, 1024u, 1024u));
    auto output = FFT::createNewOutput(input);
    auto outSlice = foobar::types::makeSliceView<0>(foobar::getFullData(output), foobar::types::makeRange());
    auto fft = foobar::makeFFT<FFT_LIB, false>(input);

    genData(input, dataSet);
    fft(input);
    libTiff::FloatImage<> img(filePath, 1024u, 1024u);
    auto acc1 = foobar::accessors::makeTransformAccessorFor(foobar::policies::CalcIntensityFunc(), outSlice);
    foobar::accessors::TransposeAccessor<decltype(acc1)> acc(acc1);
    foobar::policies::copy(outSlice, img, acc1);
    img.save();
}

int
main(int argc, char** argv)
{
    unsigned dataSet;
    string inFilePath, outFilePath;
    bool inOrOut;
    desc.add_options()
        ("help,h", "Show help message")
        ("outputFile,o", po::value<string>(&outFilePath)->default_value("output.tif"), "Output file to write to")
        ("datasSet", po::value<unsigned>(&dataSet)->default_value(1), "Data set to use (1-3)")
        ("inOrOut, i", po::value<bool>(&inOrOut)->default_value(true), "Write Input(1) or Output(0)")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || outFilePath.empty() || dataSet < 1 || dataSet > 3 )
    {
        showHelp();
        return 1;
    }

    if(inOrOut)
        writeInput(outFilePath, dataSet);
    else
        writeFFT(outFilePath, dataSet);
}
