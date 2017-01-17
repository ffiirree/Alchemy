# --------------------------------------------------------------
#  Root CMake file for zMatrix
#
#     From the off-tree build directory, invoke:
#       $ cmake <PATH_TO_ZMATRIX_ROOT>
# --------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME zMatrix)

project(${PROJECT_NAME})

# 编译要求，使用C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 子目录
add_subdirectory(modules)

if (WIN32)
	# 头文件目录，OpenCV + Kinect SDK 2.0
	set(OPENCV_INCLUDE "$ENV{OPENCV_DIR}\\build\\include;$ENV{OPENCV_DIR}\\build\\include\\opencv2")
	#set(KINECT_INCLUDE "$ENV{KINECTSDK20_DIR}/inc")

	include_directories(${OPENCV_INCLUDE})
	#include_directories(${KINECT_INCLUDE})
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/modules)

# 生成可执行文件zMatrix
aux_source_directory(. DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${PROJECT_NAME}  zgui zimgproc zmatch zkinect)

# OpenCV
if (UNIX)
    message("This is Unix.")
    find_package(OpenCV REQUIRED)
    target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} zgui zimgproc zmatch)
elseif (WIN32)
	message("This is Windows.")
	
	# OpenCV
	set(OPENCV_LIB_DIRS $ENV{OPENCV_DIR}/build/x86/vc12/lib)
	
	MACRO(LL LIB_BASE_NAME)
		target_link_libraries(${PROJECT_NAME} debug ${OPENCV_LIB_DIRS}/${LIB_BASE_NAME}d.lib)
		target_link_libraries(${PROJECT_NAME} optimized ${OPENCV_LIB_DIRS}/${LIB_BASE_NAME}.lib)
	ENDMACRO()

	# OpenCV_LIBS
	LL(opencv_core2413)
	LL(opencv_features2d2413)
	LL(opencv_gpu2413)
	LL(opencv_highgui2413)
	LL(opencv_imgproc2413)
	LL(opencv_legacy2413)
	LL(opencv_ml2413)
	LL(opencv_flann2413)
	LL(opencv_contrib2413)
	LL(opencv_calib3d2413)
	LL(opencv_videostab2413)
	LL(opencv_superres2413)
	LL(opencv_stitching2413)
	LL(opencv_photo2413)
	LL(opencv_ocl2413)
	LL(opencv_nonfree2413)
	LL(opencv_video2413)
	LL(opencv_ts2413)
	LL(opencv_objdetect2413)

	# Kinect
	#set(KINECT_LIB_DIRS $ENV{KINECTSDK20_DIR}/Lib/x86)
	#set(KINECT_LIBS  "${KINECT_LIB_DIRS}/Kinect20.lib")
	#target_link_libraries(${PROJECT_NAME} ${KINECT_LIBS})

	# Libjpeg
	set(LIBJPEG_LIB_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib)
	set(LIBJPEG_LIBS "${LIBJPEG_LIB_DIRS}/libjpeg.lib")
	target_link_libraries(${PROJECT_NAME} ${LIBJPEG_LIBS})
endif()