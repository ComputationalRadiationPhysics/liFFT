#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "libTiff/image.hpp"
#include "libTiff/traitsAndPolicies.hpp"
#include "foobar/FFT.hpp"
#if defined(WITH_CUDA) and false // Cuda memory is not enough
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

string
replace(string str, const string& from, const string& to) {
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

void
do2D_FFT(const string& inFilePath, const string& outFilePath)
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

/**
 * Copies data from an image to a 3D container using the slice at the specified z-offset
 * @param img  Image to copy
 * @param data Datacontainer to copy to
 * @param zDim z-Index
 * @param acc  Accessor to use to access the data
 */
template< class T_Img, class T_Data, class T_Acc = foobar::traits::DefaultAccessor_t<T_Data> >
void
copy2Data(const T_Img& img, T_Data& data, unsigned zDim, const T_Acc& acc = T_Acc())
{
    foobar::types::Vec<3> idx(zDim, 0u, 0u);
    for(; idx[1] < img.getHeight(); ++idx[1])
        for(; idx[2] < img.getWidth(); ++idx[2])
            acc(idx, data) = img(idx[2], idx[1]);
}

/**
 * Copies data from a 3D container to an image using the slice at the specified z-offset
 * @param data Datacontainer to copy from
 * @param img  Image to copy to
 * @param zDim z-Index
 * @param acc  Accessor to use to access the data
 */
template< class T_Img, class T_Data, class T_Acc = foobar::traits::DefaultAccessor_t<T_Data> >
void
copy2Img(const T_Data& data, T_Img& img, unsigned zDim,  const T_Acc& acc = T_Acc())
{
    foobar::types::Vec<3> idx(zDim, 0u, 0u);
    for(; idx[1] < img.getHeight(); ++idx[1])
        for(; idx[2] < img.getWidth(); ++idx[2])
            img(idx[2], idx[1]) = acc(idx, data);
}

string
getFilledNumber(unsigned num, unsigned minSize, char filler)
{
    string s(std::to_string(num));
    while(s.size()<minSize)
        s = filler + s;
    return s;
}

int
main(int argc, char** argv)
{
    unsigned firstIdx, lastIdx, minSize;
    char filler;
    string inFilePath, outFilePath;
    desc.add_options()
        ("help,h", "Show help message")
        ("input-file,i", po::value<string>(&inFilePath), "Input file to use, can contain %i as a placeholder for 3D FFTs")
        ("output-file,o", po::value<string>(&outFilePath)->default_value("output.tif"), "Output file to write to")
        ("firstIdx", po::value<unsigned>(&firstIdx)->default_value(0), "first index to use")
        ("lastIdx", po::value<unsigned>(&lastIdx)->default_value(0), "last index to use")
        ("minSize,s", po::value<unsigned>(&minSize)->default_value(0), "Minimum size of the string replaced for %i")
        ("fillChar,f", po::value<char>(&filler)->default_value('0'), "Char used to fill the string to the minimum size")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || !vm.count("input-file") || inFilePath.empty() || outFilePath.empty())
    {
        showHelp();
        return 1;
    }

    if(firstIdx > lastIdx)
    {
        std::cerr << "FirstIdx must not be greater then lastIdx" << std::endl;
        return 1;
    }

    if(firstIdx == lastIdx || inFilePath.find("%i") == string::npos)
    {
        // Only 1 image --> 2D FFT
        inFilePath = replace(inFilePath, "%i", getFilledNumber(firstIdx, minSize, filler));
        do2D_FFT(inFilePath, outFilePath);
        return 0;
    }

    // Multiple images --> 3D FFT
    // Assume all images have the same size --> load the first one to get extents and create FFT Data
    string curFilePath = replace(inFilePath, "%i", getFilledNumber(firstIdx, minSize, filler));
    libTiff::FloatImage<> img(curFilePath);
    using FFT = foobar::FFT_3D_R2C_F;
    auto input = FFT::wrapFFT_Input(
                    foobar::mem::RealContainer<3, float>(
                            foobar::types::Vec<3>(lastIdx-firstIdx+1, img.getHeight(), img.getWidth())
                    )
                 );
    auto output = FFT::getNewFFT_Output(input);
    auto fft = foobar::makeFFT<FFT_LIB, false>(input, output);
    // Now copy all the data into one memory region
    copy2Data(img, input, 0);
    for(unsigned i=firstIdx+1; i<=lastIdx; ++i)
    {
        curFilePath = replace(inFilePath, "%i", getFilledNumber(i, minSize, filler));
        img.open(curFilePath);
        copy2Data(img, input, i-firstIdx);
    }
    // Do the FFT
    fft(input, output);
    // Copy the intensities to the img and save it
    auto acc = foobar::accessors::makeTransformAccessorFor(foobar::policies::CalcIntensityFunc(), output);
    copy2Img(output, img, 0, acc);
    img.saveTo(outFilePath);

    return 0;
}
