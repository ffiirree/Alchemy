# --------------------------------------------------------------
#  Root CMake file for zMatrix
#
#     From the off-tree build directory, invoke:
#       $ cmake <PATH_TO_ZMATRIX_ROOT>
# --------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)

project(zMatrix)

# 编译要求，使用C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 子目录
add_subdirectory(modules)

if (WIN32)
	# 头文件目录，OpenCV + Kinect SDK 2.0
	set(OPENCV_INCLUDE "$ENV{OPENCV_DIR}\\build\\include;$ENV{OPENCV_DIR}\\build\\include\\opencv2")
	set(KINECT_INCLUDE "$ENV{KINECTSDK20_DIR}/inc")
	message(OpenCV include dir=${OPENCV_INCLUDE})
	message(Kinect include dir=${KINECT_INCLUDE})
	include_directories(${OPENCV_INCLUDE})
	include_directories(${KINECT_INCLUDE})
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/modules)

# 生成可执行文件zMatrix
aux_source_directory(. DIR_SRCS)
add_executable(zMatrix ${DIR_SRCS})

# OpenCV
if (UNIX)
    message("This is Unix.")
    find_package(OpenCV REQUIRED)
    target_link_libraries(zMatrix  ${OpenCV_LIBS} zgui zimgproc zmatch)
elseif (WIN32)
	message("This is Windows.")
	
	# 库文件目录，OpenCV + Kinect SDK 2.0
	set(OPENCV_LIB_DIRS $ENV{OPENCV_DIR}/build/x86/vc12/lib)
	set(KINECT_LIB_DIRS $ENV{KINECTSDK20_DIR}/Lib/x86)
	set(LIBJPEG_LIB_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib)
	
	set(OPENCV_LIBS "${OPENCV_LIB_DIRS}/opencv_objdetect2413.lib;${OPENCV_LIB_DIRS}/opencv_ts2413.lib;${OPENCV_LIB_DIRS}/opencv_video2413.lib"
		"${OPENCV_LIB_DIRS}/opencv_nonfree2413.lib;${OPENCV_LIB_DIRS}/opencv_ocl2413.lib;${OPENCV_LIB_DIRS}/opencv_photo2413.lib;${OPENCV_LIB_DIRS}/opencv_stitching2413.lib"
		"${OPENCV_LIB_DIRS}/opencv_superres2413.lib;${OPENCV_LIB_DIRS}/opencv_videostab2413.lib;${OPENCV_LIB_DIRS}/opencv_calib3d2413.lib;${OPENCV_LIB_DIRS}/opencv_contrib2413.lib"
		"${OPENCV_LIB_DIRS}/opencv_core2413.lib;${OPENCV_LIB_DIRS}/opencv_features2d2413.lib;${OPENCV_LIB_DIRS}/opencv_flann2413.lib;${OPENCV_LIB_DIRS}/opencv_gpu2413.lib"
		"${OPENCV_LIB_DIRS}/opencv_highgui2413.lib;${OPENCV_LIB_DIRS}/opencv_imgproc2413.lib;${OPENCV_LIB_DIRS}/opencv_legacy2413.lib;${OPENCV_LIB_DIRS}/opencv_ml2413.lib")
	set(KINECT_LIBS  "${KINECT_LIB_DIRS}/Kinect20.lib")
	set(LIBJPEG_LIBS "${LIBJPEG_LIB_DIRS}/libjpeg.lib")
	
	target_link_libraries(zMatrix  zgui zimgproc zmatch zkinect ${OpenCV_LIBS} ${KINECT_LIBS} ${LIBJPEG_LIBS} )
endif()