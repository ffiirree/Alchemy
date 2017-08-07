# --------------------------------------------------------------
#  CMakeLists.txt for test module
# --------------------------------------------------------------

link_libraries(${LIBJPEG_LIBS} ${OpenCV_LIBS} zgui zimgproc zmath zfeatures2d)

# generate test files
# test_findContours
add_executable(test_findContours test_findContours.cpp ${HEADERS})
add_test(TEST_findContours ${EXECUTABLE_OUTPUT_PATH}/test_findContours)
set_property(TARGET test_findContours PROPERTY FOLDER "test")

# test_fft
add_executable(test_fft test_fft.cpp ${HEADERS})
add_test(TEST_FFT ${EXECUTABLE_OUTPUT_PATH}/test_fft)
set_property(TARGET test_fft PROPERTY FOLDER "test")

# zMatrix
aux_source_directory(zMatrix main.cpp)
add_executable(zMatrix main.cpp)
set_property(TARGET zMatrix PROPERTY FOLDER "test")

