#pragma once

namespace foobar {
namespace types {

    template< class T_Stream, unsigned T_numDims >
    struct StreamWrapper: T_Stream
    {
        using T_Stream::T_Stream;
        static constexpr unsigned numDims = T_numDims;
    };

}  // namespace types
}  // namespace foobar
