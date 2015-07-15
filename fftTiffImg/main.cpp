#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "libTiff/image.hpp"
#include "libTiff/traitsAndPolicies.hpp"
#include "foobar/FFT.hpp"
#ifdef WITH_CUDA
#include "foobar/libraries/cuFFT/cuFFT.hpp"
using FFT_LIB = foobar::libraries::cuFFT::CuFFT<>;
#else
#include "foobar/libraries/fftw/FFTW.hpp"
using FFT_LIB = foobar::libraries::fftw::FFTW<>;
#endif
#include "foobar/policies/Copy.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/accessors/TransformAccessor.hpp"
#include "foobar/accessors/TransposeAccessor.hpp"
#include "foobar/policies/CalcIntensityFunctor.hpp"

namespace po = boost::program_options;
using std::string;
using std::cout;
using std::cerr;

string replace(string str, const string& from, const string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == string::npos)
        return str;
    return str.replace(start_pos, from.length(), to);
}

po::options_description desc("Convert tiff images via FFT and stores the intensities in another tiff");

void showHelp()
{
    cout << desc << std::endl;
}

void do2D_FFT(const string& inFilePath, const string& outFilePath)
{
    using namespace foobar;
    using FFT = FFT_2D_R2C_F;
    auto input = FFT::wrapFFT_Input(libTiff::FloatImage<>(inFilePath, false));
    auto output = FFT::getNewFFT_Output(input);
    auto fft = makeFFT<FFT_LIB, false>(input, output);
    input.getBase().load();
    fft(input, output);
    libTiff::FloatImage<> outImg(outFilePath, input.getBase().getWidth(), input.getBase().getHeight());
    auto fullOutput = types::makeSymmetricWrapper(output, input.getExtents()[1]);
    auto transformAcc = accessors::makeTransposeAccessor(
                            accessors::makeTransformAccessorFor(policies::CalcIntensityFunc(), fullOutput)
                        );
    policies::copy(fullOutput, outImg, transformAcc);
    outImg.save();
}

int main(int argc, char** argv)
{
    unsigned firstIdx, lastIdx;
    string inFilePath, outFilePath;
    desc.add_options()
        ("help", "Show help message")
        ("input-file,i", po::value<string>(&inFilePath), "Input file to use, can contain %i as a placeholder for 3D FFTs")
        ("output-file,o", po::value<string>(&outFilePath)->default_value("output.tif"), "Output file to write to")
        ("firstIdx", po::value<unsigned>(&firstIdx)->default_value(0), "first index to use")
        ("lastIdx", po::value<unsigned>(&lastIdx)->default_value(0), "last index to use")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || !vm.count("input-file") || inFilePath.empty() || outFilePath.empty()) {
        showHelp();
        return 1;
    }

    if(firstIdx > lastIdx) {
        std::cerr << "FirstIdx must not be greater then lastIdx" << std::endl;
        return 1;
    }

    if(firstIdx == lastIdx || inFilePath.find("%i") == string::npos)
    {
        inFilePath = replace(inFilePath, "%i", std::to_string(firstIdx));
        do2D_FFT(inFilePath, outFilePath);
    }
}
