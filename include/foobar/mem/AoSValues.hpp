#pragma once

#include <memory>
#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/accessors/ArrayAccessor.hpp"

#include <cassert>

namespace foobar {
namespace mem {
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
        using IdentityAccessor = accessors::ArrayAccessor<>;

        AoSValues(): AoSValues(nullptr, 0){}
        AoSValues(Ptr data, size_t numElements): m_data(data), m_numElements(numElements){}

        void
        reset(Ptr data, size_t numElements)
        {
            assert(numElements || !data);
            m_data.reset(data);
            m_numElements = numElements;
        }

        void
        allocData(size_t numElements)
        {
            assert(numElements);
            m_data.reset(new Value[numElements]);
            m_numElements = numElements;
        }

        void
        freeData()
        {
            m_data.reset();
            m_numElements = 0;
        }

        Ptr
        releaseData()
        {
            m_numElements = 0;
            return m_data.release();
        }

        Ptr
        getData() const
        {
            return m_data.get();
        }

        size_t
        getNumElements() const
        {
            return m_numElements;
        }

        size_t
        getMemSize() const
        {
            return m_numElements * sizeof(T);
        }

        ConstRef
        operator[](size_t idx) const
        {
            assert(idx < m_numElements);
            return m_data[idx];
        }

        Ref
        operator[](size_t idx)
        {
            assert(idx < m_numElements);
            return m_data[idx];
        }

    private:
        Data m_data;
        size_t m_numElements;
    };

}  // namespace detail
}  // namespace mem
}  // namespace foobar
