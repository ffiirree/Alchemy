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
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 子目录
add_subdirectory(modules/kinect)
add_subdirectory(modules/test)
add_subdirectory(modules/zcore)
add_subdirectory(modules/zgui)
add_subdirectory(modules/zimgproc)
add_subdirectory(modules/zmatch)


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

# OpenCV
if (UNIX)
    message("This is Unix.")
    find_package(OpenCV REQUIRED)
    target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} zgui zimgproc zmatch)
elseif (WIN32)
	message("This is Windows.")
	
	# OpenCV
	set(OPENCV_LIB_DIRS $ENV{OPENCV_DIR}/build/x86/vc12/lib)
	
	# ----------------------------------------------------------------------------------------\
	# 宏和函数
	MACRO(LL EXENAME LIB_BASE_NAME)
		target_link_libraries(${EXENAME} debug ${OPENCV_LIB_DIRS}/${LIB_BASE_NAME}d.lib)
		target_link_libraries(${EXENAME} optimized ${OPENCV_LIB_DIRS}/${LIB_BASE_NAME}.lib)
	ENDMACRO()

	# OpenCV_LIBS
	function(link_opencv exe_name)
		LL(${exe_name} opencv_core2413)
		LL(${exe_name} opencv_features2d2413)
		LL(${exe_name} opencv_gpu2413)
		LL(${exe_name} opencv_highgui2413)
		LL(${exe_name} opencv_imgproc2413)
		LL(${exe_name} opencv_legacy2413)
		LL(${exe_name} opencv_ml2413)
		LL(${exe_name} opencv_flann2413)
		LL(${exe_name} opencv_contrib2413)
		LL(${exe_name} opencv_calib3d2413)
		LL(${exe_name} opencv_videostab2413)
		LL(${exe_name} opencv_superres2413)
		LL(${exe_name} opencv_stitching2413)
		LL(${exe_name} opencv_photo2413)
		LL(${exe_name} opencv_ocl2413)
		LL(${exe_name} opencv_nonfree2413)
		LL(${exe_name} opencv_video2413)
		LL(${exe_name} opencv_ts2413)
		LL(${exe_name} opencv_objdetect2413)
	endfunction()

	macro(LINK_ALL_LIBS TN)
		target_link_libraries(${TN} ${LIBJPEG_LIBS} zgui zimgproc zmatch zkinect)
		link_opencv(${TN})
	endmacro()
	
	# ----------------------------------------------------------------------------------------/

	# Kinect
	#set(KINECT_LIB_DIRS $ENV{KINECTSDK20_DIR}/Lib/x86)
	#set(KINECT_LIBS  "${KINECT_LIB_DIRS}/Kinect20.lib")
	#target_link_libraries(${PROJECT_NAME} ${KINECT_LIBS})

	# Libjpeg libs
	set(LIBJPEG_LIB_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib)
	set(LIBJPEG_LIBS "${LIBJPEG_LIB_DIRS}/libjpeg.lib")
	
	LINK_ALL_LIBS(${PROJECT_NAME})

	# generate test files
	add_executable(test_findContours modules/test/test_findContours.cpp)
	LINK_ALL_LIBS(test_findContours)

	add_executable(test_fft modules/test/test_fft.cpp)
	LINK_ALL_LIBS(test_fft)

	# install
	set(INSTALL_DIR "D:/zMatrix")
	# header files
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/kinect/ DESTINATION "${INSTALL_DIR}/include/kinect" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/test/ DESTINATION "${INSTALL_DIR}/include/test" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zcore/ DESTINATION "${INSTALL_DIR}/include/zcore" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zgui/ DESTINATION "${INSTALL_DIR}/include/zgui" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zimgproc/ DESTINATION "${INSTALL_DIR}/include/zimgproc" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zmatch/ DESTINATION "${INSTALL_DIR}/include/zmatch" FILES_MATCHING PATTERN "*.h")

	install(FILES  ${PROJECT_SOURCE_DIR}/resources/test.jpeg DESTINATION "${INSTALL_DIR}/samples")

	# test
	enable_testing()

	# test findContours
	add_test(TEST_findContours ${EXECUTABLE_OUTPUT_PATH}/test_findContours)
	add_test(TEST_FFT ${EXECUTABLE_OUTPUT_PATH}/test_fft)

endif()