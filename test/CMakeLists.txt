find_package (Boost COMPONENTS unit_test_framework REQUIRED)

add_definitions(-DBOOST_TEST_DYN_LINK)

add_executable(minIncludes minIncludes.cpp)

set(LiFFT_ENABLE_COVERAGE OFF CACHE BOOL "Generate coverage build")
if(LiFFT_ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -coverage") # -fprofile-arcs -ftest-coverage
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CUDA_FOUND AND LiFFT_ENABLE_CUDA)
    file(GLOB LiFFT_Test_FILES
        "*.cu"
        "*.hpp"
    )
    cuda_add_executable(Test ${LiFFT_Test_FILES})
    CUDA_ADD_CUFFT_TO_TARGET(Test)
else()
    file(GLOB LiFFT_Test_FILES
        "*.cpp"
        "*.hpp"
    )
    list(REMOVE_ITEM LiFFT_Test_FILES ${CMAKE_CURRENT_SOURCE_DIR}/minIncludes.cpp)
	add_executable(Test ${LiFFT_Test_FILES})
endif()

target_link_libraries(Test ${LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_custom_command(
    TARGET Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/writeData.py"
    ${CMAKE_BINARY_DIR}
)
add_custom_command(
    TARGET Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/rect.tif"
    ${CMAKE_BINARY_DIR}
)
add_custom_command(
    TARGET Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/input1.tif"
    ${CMAKE_BINARY_DIR}
)
add_custom_command(
    TARGET Test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/input2.tif"
    ${CMAKE_BINARY_DIR}
)

