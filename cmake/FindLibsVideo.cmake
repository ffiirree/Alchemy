set(HAVE_FFMPEG OFF)
set(USE_FFMPEG OFF)

if(WIN32)
elseif(UNIX)
    # FFmpeg
    find_package(FFMPEG REQUIRED)
    if(FFMPEG_FOUND)
        set(HAVE_FFMPEG ON)
        set(USE_FFMPEG ON)
        include_directories(${FFMPEG_INCLUDE_DIR})
    endif()
endif()


