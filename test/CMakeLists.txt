# --------------------------------------------------------------
#  CMakeLists.txt for test module
# --------------------------------------------------------------

file(GLOB TEST_SRCS ./test_*.cpp)
add_executable(test_alchemy ${TEST_SRCS} gtest_main.cpp)
target_link_libraries(test_alchemy ${ALCHEMY_LIBS} ${LINK_LIBS})

add_custom_target(runtest COMMAND test_alchemy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
