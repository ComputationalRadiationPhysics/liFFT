/* This file is part of libLiFFT.
 *
 * libLiFFT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libLiFFT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libLiFFT.  If not, see <www.gnu.org/licenses/>.
 */
 
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
