if(WIN32)
    set(HAVE_JPEG ON)
    set(USE_JPEG ON)
    set(JPEG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
    set(JPEG_LIBRARIES "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib/libjpeg.lib")
    include_directories(${JPEG_INCLUDE_DIR})
elseif(UNIX)

    # libjpeg
    find_package(JPEG)
    if(JPEG_FOUND)
        set(HAVE_JPEG ON)
        set(USE_JPEG ON)
        include_directories(${JPEG_INCLUDE_DIR})
    else(JPEG_FOUND)
        message(NOT FOUND JPEG)
    endif(JPEG_FOUND)

    # libpng
    find_package(PNG)
    if(PNG_FOUND)
        set(HAVE_PNG ON)
    endif()

    # libzib
    find_package(ZLIB)
    if(ZLIB_FOUND)
        set(HAVE_ZLIB ON)
    endif()
endif(WIN32)


