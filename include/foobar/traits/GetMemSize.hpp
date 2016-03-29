#pragma once

namespace foobar {
namespace traits {

    /**
     * Returns the size of the allocated memory in bytes
     */
    template<class T>
    struct GetMemSize
    {
        size_t operator()(const T& data) const
        {
            return data.getMemSize();
        }
    };

    template<class T>
    struct GetMemSize<const T>: GetMemSize<T>{};

    template<class T>
    struct GetMemSize<T&>: GetMemSize<T>{};

    template<class T>
    size_t
    getMemSize(const T& data)
    {
        return GetMemSize<T>()(data);
    }

}  // namespace traits
}  // namespace foobar
