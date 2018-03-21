if(WIN32)
    # libjpeg
    set(JPEG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
    set(JPEG_LIBRARIES "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib/libjpeg.lib")
    
    set(HAVE_JPEG ON)
    set(USE_JPEG ON)
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
    set(USE_FFTW ON)
elseif(UNIX)
    # libjpeg
    find_package(JPEG)
    if(JPEG_FOUND)
        set(HAVE_JPEG ON)
        set(USE_JPEG ON)

        include_directories(${JPEG_INCLUDE_DIR})
    else(JPEG_FOUND)
    set(USE_JPEG OFF)
    endif(JPEG_FOUND)

    # FFTW3
    find_package(FFTW)
    if(FFTW_FOUND)
        set(HAVE_FFTW ON)
        set(USE_FFTW ON)
        add_definitions(-DUSE_FFTW)

        include_directories(${FFTW_INCLUDES})
    else()
        set(USE_FFTW OFF)
    endif()

     # libzib
     find_package(ZLIB)
     if(ZLIB_FOUND)
         set(HAVE_ZLIB ON)
         set(USE_ZLIB ON)
         include_directories(${ZLIB_INCLUDE_DIR})
     else()
         set(HAVE_ZLIB OFF)
     endif()

     # libpng
     find_package(PNG)
     if(PNG_FOUND)
         set(HAVE_PNG ON)
         set(USE_PNG ON)

         include_directories(${PNG_INCLUDE_DIR})
     else()
         set(HAVE_PNG OFF)
     endif()


endif(WIN32)


