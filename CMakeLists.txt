# Hack for C++11 Support in eclipse
# https://public.kitware.com/Bug/bug_relationship_graph.php?bug_id=15316&graph=relation
if (${CMAKE_EXTRA_GENERATOR} MATCHES "Eclipse CDT4")
    set(CMAKE_CXX_COMPILER_ARG1 "-std=c++11" CACHE STRING "C++ version for eclipse" FORCE)
    set(CMAKE_ECLIPSE_VERSION "4.4" CACHE STRING "Eclipse version" FORCE)
endif ()

project(haLT)
cmake_minimum_required(VERSION 2.8.0)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{HOME}/Software/tiff $ENV{HOME}/Software/FFTW)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

###############################################################################
# TIFF
###############################################################################
find_package(TIFF REQUIRED)
set(LIBS ${LIBS} ${TIFF_LIBRARIES})
include_directories(${TIFF_INCLUDE_DIR})

###############################################################################
# FFTW
###############################################################################
find_package(FFTW REQUIRED)
set(LIBS ${LIBS} ${FFTW_LIB} ${FFTW_F_LIB})
include_directories(${FFTW_INCLUDES})

###############################################################################
# CUDA
###############################################################################
find_package(CUDA)
set(ENABLE_CUDA ${CUDA_FOUND} CACHE BOOL "Use CUDA")
###############################################################################
# Boost
###############################################################################
find_package(Boost 1.48.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -Xcompiler -Wall")
list(APPEND CUDA_NVCC_FLAGS_RELEASE "-O3 -DNDEBUG")
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-O0 -g")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CUDA_FOUND AND ENABLE_CUDA)
    list(APPEND CUDA_NVCC_FLAGS "-DWITH_CUDA")
	cuda_add_executable(haLT_Test main.cu)
	CUDA_ADD_CUFFT_TO_TARGET(haLT_Test)
else()
	add_executable(haLT_Test main.cpp)
endif()

add_executable(tiff2PDF tiff2pdf.cpp)

target_link_libraries(haLT_Test ${LIBS})
target_link_libraries(tiff2PDF ${TIFF_LIBRARIES})

add_custom_command(
    TARGET haLT_Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/writeData.py"
    $<TARGET_FILE_DIR:haLT_Test>
)
add_custom_command(
    TARGET haLT_Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/formFactor.py"
    $<TARGET_FILE_DIR:haLT_Test>
)
add_custom_command(
    TARGET haLT_Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/helpModule.py"
    $<TARGET_FILE_DIR:haLT_Test>
)
add_custom_command(
    TARGET haLT_Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/rect.tif"
    $<TARGET_FILE_DIR:haLT_Test>
)
