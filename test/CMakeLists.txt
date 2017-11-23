# --------------------------------------------------------------
#  CMakeLists.txt for test module
# --------------------------------------------------------------

link_libraries(zcore zgui zimgproc zmath zfeatures2d ${FFTW_LIBRARIES} ${JPEG_LIBRARIES} ${FFMPEG_LIBRARIES} ${OpenCV_LIBS} ${GTK2_LIBRARIES})

# test_new_zmatrix
add_executable(test_new_zmatrix test_new_zmatrix.cpp)
add_test(TEST_NEW_ZMATRIX ${EXECUTABLE_OUTPUT_PATH}/test_new_zmatrix)
set_property(TARGET test_new_zmatrix PROPERTY FOLDER "test")

# test_findContours
add_executable(test_findContours test_findContours.cpp ${HEADERS})
add_test(TEST_findContours ${EXECUTABLE_OUTPUT_PATH}/test_findContours)
set_property(TARGET test_findContours PROPERTY FOLDER "test")

# test_fft
add_executable(test_fft test_fft.cpp ${HEADERS})
add_test(TEST_FFT ${EXECUTABLE_OUTPUT_PATH}/test_fft)
set_property(TARGET test_fft PROPERTY FOLDER "test")


# zMatrix
add_executable(zMatrix main.cpp)
add_test(TEST_ZMATRIX ${EXECUTABLE_OUTPUT_PATH}/zMatrix)
set_property(TARGET zMatrix PROPERTY FOLDER "test")

