set(HAVE_FFMPEG OFF)
set(USE_FFMPEG OFF)

if(WIN32)
    set(HAVE_FFMPEG ON)
    set(USE_FFMPEG ON)

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
    # FFmpeg
    find_package(FFMPEG REQUIRED)
    if(FFMPEG_FOUND)
        set(HAVE_FFMPEG ON)
        set(USE_FFMPEG ON)
        include_directories(${FFMPEG_INCLUDE_DIR})
    endif()
endif()


