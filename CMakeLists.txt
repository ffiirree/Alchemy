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

# 子目录
add_subdirectory(modules/kinect)
add_subdirectory(test)
add_subdirectory(modules/zcore)
add_subdirectory(modules/zgui)
add_subdirectory(modules/zimgproc)
add_subdirectory(modules/zmatch)


find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	message("## OpenCV: Found")
	include_directories(${PROJECT_SOURCE_DIR})
	include_directories(${PROJECT_SOURCE_DIR}/modules)

	include_directories(${OpenCV_INCLUDE_DIRS})
	set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} CACHE INTERNAL "OpenCV include dirs")
	# 生成可执行文件zMatrix
	aux_source_directory(. DIR_SRCS)
	add_executable(${PROJECT_NAME} ${DIR_SRCS})

	set(LIBJPEG_LIB_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib)
	set(LIBJPEG_LIBS "${LIBJPEG_LIB_DIRS}/libjpeg.lib")
	set(OPENCV_LIBS ${OpenCV_LIBS})
	target_link_libraries(${PROJECT_NAME} ${LIBJPEG_LIBS} ${OpenCV_LIBS} zgui zimgproc zmatch)
	set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "test")

	if(UNIX)
		message("## This is Unix.")
		set(INSTALL_DIR "D:/zMatrix")

	elseif(WIN32)
		message("## This is Windows.")

		# set(KINECT_INCLUDE_DIRS $ENV{KINECTSDK20_DIR}/inc CACHE INTERNAL "Kinect include dirs")
		# set(KINECT_LIB_DIRS $ENV{KINECTSDK20_DIR}/Lib/x86)
		# set(KINECT_LIBS  "${KINECT_LIB_DIRS}/Kinect20.lib")
		# target_link_libraries(${PROJECT_NAME} ${KINECT_LIBS})

		set(INSTALL_DIR "D:/zMatrix")
	endif()



	# install
	
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

endif(OpenCV_FOUND)