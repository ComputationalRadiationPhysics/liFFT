#pragma once

#include "foobar/types/Real.hpp"
#include "foobar/types/Complex.hpp"
#include "foobar/types/DataContainer.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/traits/IntegralType.hpp"
#include "foobar/policies/LoopNDims.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/DataContainerAccessor.hpp"
#include "foobar/c++14_types.hpp"

namespace foobar {
    namespace types {

        /**
         * A container that can load a file to an internal (contiguous) memory
         *
         * @param T_FileHandler File class. Must support open(string), isOpen(), close(), and a specialization for GetExtents
         * @param T_FileReaderPolicy Policy that should provide an operator(TFileHandler&, Data&, DataAccessor&) which
         *          should read the entire file into the DataContainer using the DataAccessor(Idx, Data, Value) method
         * @param T_Accuracy The internal datatype used (float or double) [float]
         * @param T_isComplex Whether the values are complex [false]
         * @param T_numDims number of dimensions [Number of dimensions supported by the FileHandler]
         */
        template<
            typename T_FileHandler,
            typename T_FileReaderPolicy,
            typename T_Accuracy = float,
            bool T_isComplex = false,
            unsigned T_numDims = traits::NumDims< T_FileHandler >::value
        >
        class FileContainer
        {
        public:
            using FileHandler = T_FileHandler;
            using FileReaderPolicy = T_FileReaderPolicy;
            using DataAccessor = policies::DataContainerAccessor;
            static constexpr unsigned numDims = T_numDims;
            static constexpr bool isComplex = T_isComplex;
            using Accuracy = T_Accuracy;
        private:
            using ElementType = std::conditional_t< T_isComplex, Complex<Accuracy>, Real<Accuracy> >;
            using Memory = ElementType*;

            using Data = DataContainer< numDims, Memory >;
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
                if(data_.data)
                    return;
                assert(fileHandler_.isOpen());
                unsigned numEl = policies::GetNumElements< Data >()(data_);
                data_.data = static_cast<Memory>(malloc(sizeof(ElementType) * numEl));
            }

            void freeData()
            {
                free(data_.data);
                data_.data = nullptr;
            }

        public:
            FileContainer(): FileContainer(""){}
            explicit FileContainer(const std::string& filePath)
            {
                data_.data = nullptr;
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

            Memory
            getAllocatedMemory()
            {
                allocData();
                return data_.data;
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
                FileReaderPolicy()(fileHandler_, data_, DataAccessor());
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
