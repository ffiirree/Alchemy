# --------------------------------------------------------------
#  Root CMake file for zMatrix
#
#     From the off-tree build directory, invoke:
#       $ cmake <PATH_TO_ZMATRIX_ROOT>
# --------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME zMatrix)
project(${PROJECT_NAME})

# 使用项目分组
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# 编译要求，使用C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message("## Cxx Flags: " ${CMAKE_CXX_FLAGS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 安装目录
if(UNIX)
	message("## This is Unix.")
	set(INSTALL_DIR "/usr/zMatrix")
elseif(WIN32)
	message("## This is Windows.")
	set(INSTALL_DIR "D:/zMatrix")
endif()

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	# libjpeg
	set(LIBJPEG_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
	set(LIBJPEG_LIBS "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib/libjpeg.lib")

	include_directories(${PROJECT_SOURCE_DIR}/modules)
	include_directories(${OpenCV_INCLUDE_DIRS})
	include_directories(${LIBJPEG_INCLUDE_DIRS})

	# 生成不同版本的库
	set(CMAKE_DEBUG_POSTFIX "d")

	# 子目录
	add_subdirectory(modules/kinect)
	add_subdirectory(modules/zcore)
	add_subdirectory(modules/zgui)
	add_subdirectory(modules/zimgproc)
	add_subdirectory(modules/zfeatures2d)
	add_subdirectory(modules/zmatch)

	enable_testing()
	add_subdirectory(test)

	# install
	# header files
	install(FILES  ${PROJECT_SOURCE_DIR}/modules/zmatrix.h DESTINATION "${INSTALL_DIR}/include/")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/kinect/ DESTINATION "${INSTALL_DIR}/include/kinect" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zcore/ DESTINATION "${INSTALL_DIR}/include/zcore" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zgui/ DESTINATION "${INSTALL_DIR}/include/zgui" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zimgproc/ DESTINATION "${INSTALL_DIR}/include/zimgproc" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zfeatures2d/ DESTINATION "${INSTALL_DIR}/include/zfeatures2d" FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zmatch/ DESTINATION "${INSTALL_DIR}/include/zmatch" FILES_MATCHING PATTERN "*.h")

endif(OpenCV_FOUND)





