include_directories(${PROJECT_SOURCE_DIR}/modules)


file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})

    enable_testing()

    # generate test files
    add_executable(test_findContours test_findContours.cpp ${HEADERS})
    target_link_libraries(test_findContours ${LIBJPEG_LIBS} ${OpenCV_LIBS} zgui zimgproc zmatch)
    add_test(TEST_findContours ${EXECUTABLE_OUTPUT_PATH}/test_findContours)
    set_property(TARGET test_findContours PROPERTY FOLDER "test")


    add_executable(test_fft test_fft.cpp ${HEADERS})
    target_link_libraries(test_fft ${LIBJPEG_LIBS} ${OpenCV_LIBS} zgui zimgproc zmatch)
    add_test(TEST_FFT ${EXECUTABLE_OUTPUT_PATH}/test_fft)
    set_property(TARGET test_fft PROPERTY FOLDER "test")
endif()