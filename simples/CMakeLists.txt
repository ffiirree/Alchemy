# --------------------------------------------------------------
#  CMakeLists.txt for simples module
# --------------------------------------------------------------

function(add_simples name)
    if (USE_CUDA AND CUDA_FOUND)
        set_source_files_properties(${name}.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
        cuda_add_executable(${name} ${name}.cpp)
    else()
        add_executable(${name} ${name}.cpp)
    endif()
    target_link_libraries(${name} ${ALCHEMY_LIBS} ${LINK_LIBS})
endfunction()

add_simples(MatrixOperations)
add_simples(MatrixThreshold)
add_simples(MatrixROI)

add_simples(VideoCapture)
add_simples(Filters)
add_simples(FindContours)
add_simples(Morp)

add_simples(Net_01)
add_simples(Net_02_cross_entropy)
add_simples(Net_03_tanh)
add_simples(Net_04_softmax)
add_simples(Net_05_RELU)
add_simples(Net_06_CNN_CPU_ONLY)
add_simples(Net_07_dropout)
add_simples(Net_08_CNN_CuDNN)
add_simples(LeNet)
add_simples(Image_split_merge)