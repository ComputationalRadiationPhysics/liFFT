#pragma once

#include <memory>
#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/policies/ArrayAccessor.hpp"

namespace foobar {
namespace types {
namespace detail {

    /**
     * Deleter that does nothing
     */
    struct NopDeleter
    {
        template< typename T >
        void
        operator()(T){}
    };

    /**
     * Wrapper to hold and manage an Array of Structs
     *
     * \tparam T Type to hold
     * \tparam T_ownsPointer Whether this class owns its pointer or not (memory is freed on destroy, when true)
     */
    template< typename T, bool T_ownsPointer = true >
    class AoSValues
    {
    public:
        using type = typename traits::IntegralType<T>::type;
        static constexpr bool ownsPointer = T_ownsPointer;
        static constexpr bool isComplex = traits::IsComplex<T>::value;
        static constexpr bool isAoS = true;
        using Value = T;
        using Ptr = Value*;
        using Ref = Value&;
        using ConstRef = const Value&;
        using Data = std::conditional_t<
                        ownsPointer,
                        std::unique_ptr< Value[] >,
                        std::unique_ptr< Value[], NopDeleter >
                     >;
        using Accessor = policies::ArrayAccessor<>;

        AoSValues(): data_(nullptr){}
        AoSValues(Ptr data): data_(data){}

        void
        operator=(Ptr data)
        {
            data_.reset(data);
        }

        void
        allocData(size_t numElements)
        {
            data_.reset(new Value[numElements]);
        }

        void
        freeData()
        {
            data_.reset();
        }

        void
        releaseData()
        {
            data_.release();
        }

        Ptr
        getData() const
        {
            return data_.get();
        }

        ConstRef
        operator[](size_t idx) const
        {
            return data_[idx];
        }

        Ref
        operator[](size_t idx)
        {
            return data_[idx];
        }

    private:
        Data data_;
    };

}  // namespace detail
}  // namespace types
}  // namespace foobar
