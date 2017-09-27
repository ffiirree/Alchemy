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
set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 安装目录
if(UNIX)
	set(INSTALL_DIR "/usr/zMatrix")
elseif(WIN32)
	set(INSTALL_DIR "D:/zMatrix")
endif()


option(USE_FFMPEG 	"Include FFMPEG support" ON)
option(USE_LIBJPEG 	"Include FFMPEG support" ON)
option(USE_WIN32 	"Include FFMPEG support" ON)
option(USE_OPENCV 	"Include FFMPEG support" ON)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	# libjpeg
	set(LIBJPEG_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
	set(LIBJPEG_LIBS "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib/libjpeg.lib")

	# fftw
	set(FFTW_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/fftw")
	set(FFTW_LIBS 
		"${PROJECT_SOURCE_DIR}/3rdparty/fftw/libfftw3-3.lib"
		"${PROJECT_SOURCE_DIR}/3rdparty/fftw/libfftw3f-3.lib" 
		"${PROJECT_SOURCE_DIR}/3rdparty/fftw/libfftw3l-3.lib")

	include_directories(${PROJECT_SOURCE_DIR}/modules)
	include_directories(${OpenCV_INCLUDE_DIRS})
	include_directories(${LIBJPEG_INCLUDE_DIRS})
	include_directories(${FFTW_INCLUDE_DIRS})

	# 生成不同版本的库
	set(CMAKE_DEBUG_POSTFIX "d")

	# 子目录
	add_subdirectory(modules/kinect)
	add_subdirectory(modules/zcore)
	add_subdirectory(modules/zgui)
	add_subdirectory(modules/zimgproc)
	add_subdirectory(modules/zfeatures2d)
	add_subdirectory(modules/zmath)

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
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zmath/ DESTINATION "${INSTALL_DIR}/include/zmath" FILES_MATCHING PATTERN "*.h")

endif(OpenCV_FOUND)

# log
function(log text)
	set(status_cond)

	foreach(arg ${ARGN})
		list(APPEND status_cond ${arg})
	endforeach()

	set(status_placeholder_length 32)
	string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
	string(LENGTH "${text}" status_text_length)
	if(status_text_length LESS status_placeholder_length)
	  	string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
	else()
	  	set(status_text "${text}")
	endif()

	string(REPLACE ";" " " status_cond "${status_cond}")
  	string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
  	message(STATUS "~ ${status_text} ${status_cond}")
endfunction()

# ========================== build platform ==========================
log("")
log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Platform ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log("")
log("    Host:"             ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
log("    CMake:" ${CMAKE_VERSION})
log("    CMake generator:" ${CMAKE_GENERATOR})
log("    CMake build tool:" ${CMAKE_BUILD_TOOL})
if(MSVC)
	log("    MSVC:" ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
	log("    Xcode:" ${XCODE_VERSION})
endif()	
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
	log("    Configuration:" ${CMAKE_BUILD_TYPE})
endif()


# ========================== C/C++ options ==========================
log("")
log("")
log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ C/C++ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log("")
if(NOT BUILD_SHARED_LIBS)
	log("    Built as dynamic libs?:" NO)
elseif()
	log("    Built as dynamic libs?:" YES)
endif()
log("    C++11:" YES)
log("    C++ Compiler:"           "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION})")
log("    C++ flags (Release):"    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
log("    C++ flags (Debug):"      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
log("    C Compiler:"             ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1})
log("    C flags (Release):"      ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
log("    C flags (Debug):"        ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
if(WIN32)
	log("    Linker flags (Release):" ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
	log("    Linker flags (Debug):"   ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
else()
	log("    Linker flags (Release):" ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
	log("    Linker flags (Debug):"   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()


if(HAVE_WIN32UI)
log("    Win32 UI:"  YES)
endif()

log("")