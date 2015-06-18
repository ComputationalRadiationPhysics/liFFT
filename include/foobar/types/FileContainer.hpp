#pragma once

#include "foobar/types/RealValues.hpp"
#include "foobar/types/ComplexAoSValues.hpp"
#include "foobar/types/DataContainer.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/traits/IntegralType.hpp"
#include "foobar/policies/LoopNDims.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/DataContainerAccessor.hpp"
#include "foobar/c++14_types.hpp"

#include <boost/mpl/apply.hpp>
#include "foobar/policies/Copy.hpp"

namespace bmpl = boost::mpl;

namespace foobar {
    namespace types {

        /**
         * A container that can load a file to an internal (contiguous) memory
         *
         * \tparam T_FileHandler File class. Must support open(string), isOpen(), close(), and a specialization for GetExtents
         * \tparam T_FileAccessor Either an Array- or StreamAccessor that should provide an operator([index,] TFileHandler&) which
         *          gets an element from the file
         * \tparam T_Accuracy The internal datatype used (float or double) [float]
         * \tparam T_isComplex Whether the values are complex [false]
         * \tparam T_numDims number of dimensions [Number of dimensions supported by the FileHandler]
         */
        template<
            typename T_FileHandler,
            typename T_FileAccessor,
            typename T_Accuracy = float,
            bool T_isComplex = false,
            unsigned T_numDims = traits::NumDims< T_FileHandler >::value
        >
        class FileContainer
        {
        public:
            using FileHandler = T_FileHandler;
            using DataAccessor = policies::DataContainerAccessor;
            using FileAccessor = T_FileAccessor;
            static constexpr unsigned numDims = T_numDims;
            static constexpr bool isComplex = T_isComplex;
            using Accuracy = T_Accuracy;
        private:
            using CopyPolicy = policies::Copy< FileAccessor, DataAccessor >;
            using ArrayType = std::conditional_t< isComplex, ComplexAoSValues<Accuracy>, RealValues<Accuracy> >;
            using ElementType = typename ArrayType::Value;
            using Ptr = ElementType*;

            using Data = DataContainer< numDims, ArrayType >;
            using ExtentsVec = decltype(std::declval<Data>().extents);

            FileHandler fileHandler_;
            Data data_;
            std::string filePath_;
            bool gotData_;

            void
            loadExtents()
            {
                assert(fileHandler_.isOpen());
                policies::GetExtents< FileHandler > extents(fileHandler_);
                for(unsigned i=0; i<numDims; ++i)
                    data_.extents[i] = extents[i];
            }

            void allocData()
            {
                if(data_.data.getData())
                    return;
                assert(fileHandler_.isOpen());
                unsigned numEl = policies::GetNumElements< Data >()(data_);
                data_.data.allocData(numEl);
            }

            void freeData()
            {
                data_.data.freeData();
            }

        public:
            FileContainer(): FileContainer(""){}
            explicit FileContainer(const std::string& filePath)
            {
                setFilePath(filePath);
            }

            ~FileContainer()
            {
                setFilePath("");
            }

            void setFilePath(const std::string& filePath)
            {
                if(!filePath_.empty())
                    fileHandler_.close();
                filePath_ = filePath;
                freeData();
                gotData_ = false;
                if(!filePath.empty()){
                    fileHandler_.open(filePath);
                    if(fileHandler_.isOpen())
                        loadExtents();
                }
            }

            const ExtentsVec&
            getExtents() const
            {
                assert(fileHandler_.isOpen());
                return data_.extents;
            }

            Ptr
            getAllocatedMemory()
            {
                allocData();
                return data_.data.getData();
            }

            Data&
            getData()
            {
                loadData();
                return data_;
            }

            void
            loadData(bool forceReload = false)
            {
                if(gotData_ && !forceReload)
                    return;
                gotData_ = true;
                allocData();
                CopyPolicy()(fileHandler_, data_);
            }

        };

    }  // namespace types

    namespace traits {

        template< typename T >
        struct IntegralTypeImpl< T, void_t< typename T::Accuracy > >: IntegralType< typename T::Accuracy >{};

        template<
            typename T_FileHandler,
            typename T_FileReaderPolicy,
            typename T_Accuracy,
            bool T_isComplex,
            unsigned T_numDims
        >
        struct IsStrided< types::FileContainer< T_FileHandler, T_FileReaderPolicy, T_Accuracy, T_isComplex, T_numDims > >
        : std::integral_constant< bool, false>{};

    }  // namespace traits

    namespace policies {

        template<
            typename T_FileHandler,
            typename T_FileReaderPolicy,
            typename T_Accuracy,
            bool T_isComplex,
            unsigned T_numDims
        >
        struct GetRawPtr< types::FileContainer< T_FileHandler, T_FileReaderPolicy, T_Accuracy, T_isComplex, T_numDims > >
        {
            using type = types::FileContainer< T_FileHandler, T_FileReaderPolicy, T_Accuracy, T_isComplex, T_numDims >;

            T_Accuracy*
            operator()(type& data)
            {
                return reinterpret_cast<T_Accuracy*>(data.getAllocatedMemory());
            }
        };

        template<
            typename T_FileHandler,
            typename T_FileReaderPolicy,
            typename T_Accuracy,
            bool T_isComplex,
            unsigned T_numDims
        >
        struct GetExtents< types::FileContainer< T_FileHandler, T_FileReaderPolicy, T_Accuracy, T_isComplex, T_numDims > >
        {
            using type = types::FileContainer< T_FileHandler, T_FileReaderPolicy, T_Accuracy, T_isComplex, T_numDims >;

            GetExtents(const type& data): extents_(data.getExtents()){}

            unsigned
            operator[](unsigned dim) const
            {
                return extents_[dim];
            }
        private:
            const types::Vec<2>& extents_;
        };

    }  // namespace policies

}  // namespace foobar
