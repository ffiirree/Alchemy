find_package(OpenCV)
if(OpenCV_FOUND AND USE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()