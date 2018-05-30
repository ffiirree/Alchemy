if(WIN32)
#    ADD_DEFINITIONS(-DUSE_WIN32)
#
#    # libjpeg
#    set(JPEG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
#    set(JPEG_LIBRARIES "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib/libjpeg.lib")
#
#    include_directories(${JPEG_INCLUDE_DIR})
#
#    # fftw3
#    set(FFTW_ROOT_DIR "${PROJECT_SOURCE_DIR}/3rdparty/fftw")
#    set(FFTW_INCLUDES ${FFTW_ROOT_DIR})
#    set(FFTW_LIBRARIES  "${FFTW_ROOT_DIR}/libfftw3-3.lib" "${FFTW_ROOT_DIR}/libfftw3f-3.lib" "${FFTW_ROOT_DIR}/libfftw3l-3.lib")
#    add_definitions(-DUSE_FFTW)
#
#    # Copy *.dll files
#    file(GLOB FFTW_DLIBS "${FFTW_ROOT_DIR}/*.dll")
#    add_custom_command(
#            TARGET COPY_DLL_FILES
#            POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E make_directory "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/"
#            COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${FFTW_DLIBS} "${PROJECT_SOURCE_DIR}/bin/$<CONFIG>/"
#    )
#
#    # headers
#    include_directories(${FFTW_INCLUDES})
#
#    # FFMPEG
#    set(FFMPEG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/include")
#    set(FFMPEG_LIBRARIES
#            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avcodec-58.lib"
#            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avdevice-58.lib"
#            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avformat-58.lib"
#            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avutil-56.lib"
#            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/swscale-5.lib"
#            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avfilter-7.lib"
#            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/postproc-55.lib"
#            )
#
#    # Copy *.dll files
#    file(GLOB FFMPEG_DLIBS "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/bin/*.dll")
#    add_custom_command(
#            TARGET COPY_DLL_FILES
#            POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E make_directory "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/"
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLIBS} "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/"
#    )
#
#    include_directories(${FFMPEG_INCLUDE_DIR})
elseif(UNIX)
    find_package(GTK2 REQUIRED)
    ADD_DEFINITIONS(-DUSE_GTK2)
    include_directories(${GTK2_INCLUDE_DIRS})
    list(APPEND LINK_LIBS ${GTK2_LIBRARIES})

    # libjpeg
    find_package(JPEG REQUIRED)
    include_directories(${JPEG_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${JPEG_LIBRARIES})

    # FFTW3
    find_package(FFTW REQUIRED)
    add_definitions(-DUSE_FFTW)
    include_directories(${FFTW_INCLUDES})
    list(APPEND LINK_LIBS ${FFTW_LIBRARIES})

    # libzib
    find_package(ZLIB REQUIRED)
    include_directories(${ZLIB_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${ZLIB_LIBRARIES})

    # libpng
    find_package(PNG REQUIRED)
    include_directories(${PNG_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${PNG_LIBRARIES})

    # FFmpeg
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${FFMPEG_LIBRARIES})
endif()

find_package(OpenCV)
if(OpenCV_FOUND AND USE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND LINK_LIBS ${OpenCV_LIBS})
endif()

find_package(CUDA)
if(CUDA_FOUND AND USE_CUDA)
    add_definitions(-DUSE_CUDA)

    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND LINK_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

    list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
    set(CUDA_PROPAGATE_HOST_FLAGS ON)

    find_package(CuDNN)
    if(CUDNN_FOUND AND USE_CUDNN)
        add_definitions(-DUSE_CUDNN)
        
        include_directories(${CUDNN_INCLUDE_DIRS})
        list(APPEND LINK_LIBS ${CUDNN_LIBRARIES})
    endif()
endif()

# atblas
find_package(Atlas REQUIRED)
include_directories(${Atlas_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${Atlas_LIBRARIES})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${GLOG_LIBRARIES})

find_package(NNPACK REQUIRED)
include_directories(${NNPACK_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${NNPACK_LIBRARIES} -lpthread)