#pragma once

#include "foobar/types/Vec.hpp"
#include "foobar/traits/IdentityAccessor.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/accessors/ArrayAccessor.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Range.hpp"

namespace foobar {
namespace types {

    /**
     * Provides a view on a container
     * Outside users will see it like it has a specific extent which is only a part of the real extents
     * The view is specified by an offset (in each dimension) and the extents (in each dimension)
     */
    template<
        class T_Base,
        unsigned T_fixedDim,
        typename T_HasInstance,
        class T_BaseAccessor = traits::IdentityAccessor_t<T_Base>
    >
    class SliceView
    {
        using Base = T_Base;
        static constexpr unsigned fixedDim = T_fixedDim;
        static constexpr bool hasInstance = T_HasInstance::value;
        using BaseAccessor = T_BaseAccessor;
        static constexpr unsigned baseNumDims = traits::NumDims<Base>::value;
        static_assert(baseNumDims > 1, "Cannot remove last dimension");
        static_assert(fixedDim < baseNumDims, "Fixed dimension does not exist");

        using RefType = typename std::add_lvalue_reference<Base>::type;
        using InstanceType = std::conditional_t< hasInstance, Base, RefType >;
        using ParamType = typename std::conditional_t< hasInstance, std::add_rvalue_reference<Base>, std::add_lvalue_reference<Base> >::type;
    public:
        static constexpr unsigned numDims = baseNumDims - 1;
        using Extents = Vec<numDims>;
        using BaseExtents = Vec<baseNumDims>;

    private:
        InstanceType base_;
        BaseAccessor acc_;
        BaseExtents offsets_;
        Extents extents;

        friend class policies::GetExtents<SliceView>;

        constexpr unsigned getIdx(unsigned baseIdx) const
        {
            return (baseIdx>fixedDim) ? baseIdx-1 : baseIdx;
        }

        Extents
        removeDimFromVec(const BaseExtents& vec)
        {
            Extents res;
            for(unsigned i=0; i<baseNumDims; ++i)
            {
                if(i != fixedDim)
                    res[getIdx(i)] = vec[i];
            }
            return res;
        }
    public:
        using IdentityAccessor = accessors::ArrayAccessor<true>;

        /**
         * Creates a view with offsets and extents
         * Validation on those is performed at runtime
         *
         * @param base Base container
         * @param offsets Offsets for each dimension
         * @param extents New extents
         * @param acc Accessor to access the base class
         */
        SliceView(ParamType base, const BaseExtents& offsets, const Extents& extents, const BaseAccessor& acc = BaseAccessor()):
            base_(static_cast<ParamType>(base)), acc_(acc), offsets_(offsets), extents(extents)
        {
            policies::GetExtents<Base> bExtents(base_);
            for(unsigned i=0; i<baseNumDims; ++i)
            {
                if(offsets[i] >= bExtents[i])
                    throw std::runtime_error("Invalid offset");
                if(i != fixedDim)
                {
                    if(offsets[i] + extents[getIdx(i)] > bExtents[i])
                        throw std::runtime_error("Invalid extents");
                }
            }
        }

        /**
         * Creates a view with offsets and extents
         * Validation on those is performed at runtime
         * This accepts a full extents vector (from base type) and deletes the fixed dimension (implicitly setting it to 1)
         *
         * @param base Base container
         * @param offsets Offsets for each dimension
         * @param extents New extents
         * @param acc Accessor to access the base class
         */
        SliceView(ParamType base, const BaseExtents& offsets, const BaseExtents& extents, const BaseAccessor& acc = BaseAccessor()):
            SliceView(static_cast<ParamType>(base), offsets, removeDimFromVec(extents), acc)
        {}

        template<typename T_Idx>
        std::result_of_t< BaseAccessor(const BaseExtents&, Base&) >
        operator()(const T_Idx& idx)
        {
            static_assert(traits::NumDims<T_Idx>::value == numDims, "Wrong Idx dimensions");
            BaseExtents idxNew;
            for(unsigned i=0; i<baseNumDims; ++i)
                idxNew[i] = (i==fixedDim) ? offsets_[i] : offsets_[i] + idx[getIdx(i)];
            return acc_(idxNew, base_);
        }

        template<typename T_Idx>
        std::result_of_t< BaseAccessor(const BaseExtents&, const Base&) >
        operator()(const T_Idx& idx) const
        {
            static_assert(traits::NumDims<T_Idx>::value == numDims, "Wrong Idx dimensions");
            BaseExtents idxNew;
            for(unsigned i=0; i<baseNumDims; ++i)
                idxNew[i] = (i==fixedDim) ? offsets_[i] : offsets_[i] + idx[getIdx(i)];
            const Base& cBase = const_cast<const Base&>(base_);
            return acc_(idxNew, cBase);
        }

        /**
         * Returns a reference to the base class
         * @return Reference to base data
         */
        RefType
        getBase()
        {
            return base_;
        }

        const Extents&
        getExtents() const
        {
            return extents;
        }
    };

    template<
            unsigned T_fixedDim,
            class T_Base,
            class T_BaseAccessor = traits::IdentityAccessor_t<std::remove_reference_t<T_Base>>,
            class T_Range
        >
    SliceView< std::remove_reference_t<T_Base>, T_fixedDim, negate< std::is_lvalue_reference<T_Base> >, T_BaseAccessor >
    makeSliceView(T_Base&& base, const T_Range& range, const T_BaseAccessor& acc = T_BaseAccessor())
    {
        using Base = std::remove_cv_t<std::remove_reference_t<T_Base>>;
        return SliceView< std::remove_reference_t<T_Base>, T_fixedDim, negate< std::is_lvalue_reference<T_Base> >, T_BaseAccessor >(
                std::forward<T_Base>(base),
                GetRangeOffset<T_Range, Base>::get(range),
                GetRangeExtents<T_Range, Base>::get(range, base),
                acc);
    }

}  // namespace types
}  // namespace foobar
