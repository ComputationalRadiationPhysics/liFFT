#pragma once

#include <vector>
#include <string>

namespace foobar {
namespace policies {

    /**
     * Provides delimiters for (up to) 2D text-files
     */
    struct TextDelimiters
    {
        static constexpr unsigned numDims = 2;

        const char*
        operator[](unsigned dim) const
        {
            if(dim == 0)
                return "\n";
            else
                return " ";
        }
    };

    /**
     * Simple stream accessor that writes the value into the stream via <<-operator
     *
     * @param T_Delimiters Must provide a getDelimiter(dimension) method to return a delimiter separating values from that dimension
     */
    template< class T_Delimiters >
    struct StreamAccessor
    {
        const T_Delimiters delimiters_;

        template< class T_Stream, typename T_Value >
        void operator()(T_Stream& stream, T_Value&& value)
        {
            stream << std::forward<T_Value>(value);
        }

        const T_Delimiters&
        getDelimiters()
        {
            return delimiters_;
        }
    };

    using StringStreamAccessor = StreamAccessor< TextDelimiters >;

}  // namespace policies
}  // namespace foobar
