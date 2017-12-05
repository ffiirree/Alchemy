# --------------------------------------------------------------
#  CMakeLists.txt for test module
# --------------------------------------------------------------

set(LINK_LIBS zcore zgui zimgproc zmath zfeatures2d zml ${FFTW_LIBRARIES} ${JPEG_LIBRARIES} ${FFMPEG_LIBRARIES} ${OpenCV_LIBS} ${GTK2_LIBRARIES})

function(test name)

    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} ${LINK_LIBS})

    add_test(TEST_${name} ${EXECUTABLE_OUTPUT_PATH}/test_${name})

    # 分组
    set_property(TARGET test_${name} PROPERTY FOLDER "test")
endfunction()


test(new_zmatrix)
test(videocapture)
test(roi)
test(fft)
test(split_merge)
test(filter)
test(morp)
test(threshold)
test(operations)
test(nn)
test(lenet)

test(find_contours)

# zMatrix
add_executable(zMatrix main.cpp)
target_link_libraries(zMatrix ${LINK_LIBS})
add_test(TEST_ZMATRIX ${EXECUTABLE_OUTPUT_PATH}/zMatrix)
set_property(TARGET zMatrix PROPERTY FOLDER "test")

