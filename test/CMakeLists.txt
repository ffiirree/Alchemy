# --------------------------------------------------------------
#  CMakeLists.txt for test module
# --------------------------------------------------------------

set(LINK_LIBS alchemy_nn alchemy_gui alchemy_imgproc alchemy_util alchemy_math alchemy_core
        ${FFTW_LIBRARIES} ${JPEG_LIBRARIES} ${FFMPEG_LIBRARIES}
        ${ZLIB_LIBRARIES} ${PNG_LIBRARIES}
        ${OpenCV_LIBS} ${GTK2_LIBRARIES} ${GLOG_LIBRARIES} ${Atlas_LIBRARIES} ${NNPACK_LIBRARIES} -lpthread)

if (USE_CUDA)
    set(LINK_LIBS ${LINK_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    if(USE_CUDNN)
        set(LINK_LIBS ${LINK_LIBS} ${CUDNN_LIBRARIES})
    endif()
endif ()

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
test(atlas)
test(nnpack)
test(net_01)
test(net_02_cross_entropy)
test(net_03_tanh)
test(net_04_softmax)
test(net_05_relu)
test(net_06_cnn)
test(net_07_dropout)
test(net_08_cudnn_cnn)
test(net_09_momentum_optimizer)
test(cublas)
test(tensor)
test(lbp)
test(cp)
test(lenet)
test(find_contours)
test(conv_with_cudnn)
test(pooling_layer)
test(create_picture)
test(read_cp)
test(sand)
test(sand_cnn)
