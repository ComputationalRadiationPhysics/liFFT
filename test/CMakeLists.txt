find_package (Boost COMPONENTS unit_test_framework REQUIRED)

add_definitions(-DBOOST_TEST_DYN_LINK)

if(CUDA_FOUND AND ENABLE_CUDA)
    file(GLOB LiFFT_Test_CU_FILES
        "*.cu"
        "*.hpp"
    )
    cuda_add_executable(Test ${LiFFT_Test_CU_FILES})
    CUDA_ADD_CUFFT_TO_TARGET(Test)
else()
    file(GLOB LiFFT_Test_CPP_FILES
        "*.cpp"
        "*.hpp"
    )
    list(REMOVE_ITEM LiFFT_Test_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/minIncludes.cpp)
	add_executable(Test ${LiFFT_Test_CPP_FILES})
endif()

add_executable(minIncludes minIncludes.cpp)

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

