find_package(OpenCV)
if(OpenCV_FOUND)
    set(USE_OPENCV ON)

    include_directories(${OpenCV_INCLUDE_DIRS})
endif()