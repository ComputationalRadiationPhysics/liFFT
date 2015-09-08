#pragma once

#include <stdexcept>

namespace tiffWriter {

    struct FormatException : public std::runtime_error
    {
       using std::runtime_error::runtime_error;
    };

    struct InfoMissingException : public std::runtime_error
    {
        InfoMissingException(std::string s): std::runtime_error("Info missing: "+s){}
    };

    struct InfoWriteException : public std::runtime_error
    {
        InfoWriteException(std::string s): std::runtime_error("Could not write "+s){}
    };

}  // namespace tiffWriter
