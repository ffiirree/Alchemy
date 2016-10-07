include_directories(${PROJECT_SOURCE_DIR}/modules)
if (WIN32)
	# 头文件目录，OpenCV + Kinect SDK 2.0
	set(OPENCV_INCLUDE "$ENV{OPENCV_DIR}/build/include;$ENV{OPENCV_DIR}/build/include/opencv2")
	set(KINECT_INCLUDE "$ENV{KINECTSDK20_DIR}/inc")
	set(LIBJPEG_INC "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
	message(libjpeg_include=${LIBJPEG_INC})

	include_directories(${OPENCV_INCLUDE})
	include_directories(${KINECT_INCLUDE})
	include_directories(${LIBJPEG_INC})
endif()

aux_source_directory(. ZGUI_SRCS)
add_library(zgui STATIC ${ZGUI_SRCS})