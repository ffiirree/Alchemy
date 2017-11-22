
if (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
    set(FFMPEG_FOUND TRUE)
else (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(_FFMPEG_AVCODEC libavcodec)
        pkg_check_modules(_FFMPEG_AVFORMAT libavformat)
        pkg_check_modules(_FFMPEG_AVUTIL libavutil)
        pkg_check_modules(_FFMPEG_AVDEVICE libavdevice)
        pkg_check_modules(_FFMPEG_SWSCALE libswscale)
    endif (PKG_CONFIG_FOUND)

    find_path(FFMPEG_AVCODEC_INCLUDE_DIR
            NAMES libavcodec/avcodec.h
            PATHS ${_FFMPEG_AVCODEC_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include /sw/include
            PATH_SUFFIXES ffmpeg libav
            )

    find_library(FFMPEG_LIBAVCODEC
            NAMES avcodec
            PATHS ${_FFMPEG_AVCODEC_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib /sw/lib
            )

    find_library(FFMPEG_LIBAVFORMAT
            NAMES avformat
            PATHS ${_FFMPEG_AVFORMAT_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib /sw/lib
            )

    find_library(FFMPEG_LIBAVUTIL
            NAMES avutil
            PATHS ${_FFMPEG_AVUTIL_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib /sw/lib
            )

    find_library(FFMPEG_LIBAVDEVICE
            NAMES avdevice
            PATHS ${_FFMPEG_AVDEVICE_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib /sw/lib
            )

    find_library(FFMPEG_SWSCALE
            NAMES swscale
            PATHS ${_FFMPEG_SWSCALE_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib /sw/lib
            )

    if (FFMPEG_LIBAVCODEC AND FFMPEG_LIBAVFORMAT AND FFMPEG_LIBAVDEVICE AND FFMPEG_SWSCALE)
        set(FFMPEG_FOUND TRUE)
    endif()

    if (FFMPEG_FOUND)
        set(FFMPEG_INCLUDE_DIR ${FFMPEG_AVCODEC_INCLUDE_DIR})
        set(FFMPEG_LIBRARIES
                ${FFMPEG_LIBAVCODEC}
                ${FFMPEG_LIBAVFORMAT}
                ${FFMPEG_LIBAVUTIL}
                ${FFMPEG_LIBAVDEVICE}
                ${FFMPEG_SWSCALE}
        )
    endif (FFMPEG_FOUND)

    if (FFMPEG_FOUND)
        if (NOT FFMPEG_FIND_QUIETLY)
            message(STATUS "Found FFMPEG or Libav: ${FFMPEG_LIBRARIES}, ${FFMPEG_INCLUDE_DIR}")
        endif (NOT FFMPEG_FIND_QUIETLY)
    else (FFMPEG_FOUND)
        if (FFMPEG_FIND_REQUIRED)
            message(FATAL_ERROR "Could not find libavcodec or libavformat or libavutil")
        endif (FFMPEG_FIND_REQUIRED)
    endif (FFMPEG_FOUND)

endif (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)


