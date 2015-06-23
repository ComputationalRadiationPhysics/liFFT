#pragma once

#include <stdexcept>

namespace libTiff {

    struct FormatException : public std::runtime_error
    {
       using std::runtime_error::runtime_error;
    };

    struct InfoMissingException : public std::runtime_error
    {
        InfoMissingException(std::string s): std::runtime_error("Info missing: "+s){}
    };

}  // namespace libTiff
