if(WIN32)
    # libjpeg
    set(JPEG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
    set(JPEG_LIBRARIES "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib/libjpeg.lib")

    include_directories(${JPEG_INCLUDE_DIR})

    # fftw3
    set(FFTW_ROOT_DIR "${PROJECT_SOURCE_DIR}/3rdparty/fftw")
    set(FFTW_INCLUDES ${FFTW_ROOT_DIR})
    set(FFTW_LIBRARIES  "${FFTW_ROOT_DIR}/libfftw3-3.lib" "${FFTW_ROOT_DIR}/libfftw3f-3.lib" "${FFTW_ROOT_DIR}/libfftw3l-3.lib")
    add_definitions(-DUSE_FFTW)

    # Copy *.dll files
    file(GLOB FFTW_DLIBS "${FFTW_ROOT_DIR}/*.dll")
    add_custom_command(
        TARGET COPY_DLL_FILES
        POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E make_directory "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${FFTW_DLIBS} "${PROJECT_SOURCE_DIR}/bin/$<CONFIG>/"
    )

    # headers
    include_directories(${FFTW_INCLUDES})

    # FFMPEG
    set(FFMPEG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/include")
    set(FFMPEG_LIBRARIES
            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avcodec-58.lib"
            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avdevice-58.lib"
            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avformat-58.lib"
            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avutil-56.lib"
            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/swscale-5.lib"
            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/avfilter-7.lib"
            "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib/postproc-55.lib"
            )

    # Copy *.dll files
    file(GLOB FFMPEG_DLIBS "${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/bin/*.dll")
    add_custom_command(
            TARGET COPY_DLL_FILES
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLIBS} "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/"
    )

    include_directories(${FFMPEG_INCLUDE_DIR})
elseif(UNIX)
    # libjpeg
    find_package(JPEG REQUIRED)
    if(JPEG_FOUND)
        include_directories(${JPEG_INCLUDE_DIR})
    endif(JPEG_FOUND)

    # FFTW3
    find_package(FFTW REQUIRED)
    if(FFTW_FOUND)
        add_definitions(-DUSE_FFTW)
        include_directories(${FFTW_INCLUDES})
    endif()

     # libzib
     find_package(ZLIB REQUIRED)
     if(ZLIB_FOUND)
         include_directories(${ZLIB_INCLUDE_DIR})
     endif()

     # libpng
     find_package(PNG REQUIRED)
     if(PNG_FOUND)
         include_directories(${PNG_INCLUDE_DIR})
     endif()

    # FFmpeg
    find_package(FFMPEG REQUIRED)
    if(FFMPEG_FOUND)
        include_directories(${FFMPEG_INCLUDE_DIR})
    endif()
endif(WIN32)

find_package(OpenCV)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()