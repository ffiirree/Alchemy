# --------------------------------------------------------------
#  CMakeLists.txt for test module
# --------------------------------------------------------------

find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB TEST_SRCS ./test_*)

if (USE_CUDA AND CUDA_FOUND)
    cuda_add_executable(test_alchemy ${TEST_SRCS} gtest_main.cpp)
else()
    add_executable(test_alchemy ${TEST_SRCS} gtest_main.cpp)
endif()

#add_executable(test_alchemy ${TEST_SRCS} gtest_main.cpp)
target_link_libraries(test_alchemy ${ALCHEMY_LIBS} ${LINK_LIBS} ${GTEST_LIBRARIES})

add_custom_target(runtest COMMAND test_alchemy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
