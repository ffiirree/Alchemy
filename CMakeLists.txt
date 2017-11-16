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
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ========================== configuration ==========================
configure_file ("${PROJECT_SOURCE_DIR}/cmake/config.h.in" "${PROJECT_SOURCE_DIR}/modules/zcore/config.h")

# ========================== Options ==========================
option(USE_OPENCV 	"Include OpenCV support" 	OFF)
option(USE_KINECT 	"Include Kinect support"	OFF)
option(USE_FFTW		"Include FFTW support"		ON)
option(USE_FFMPEG 	"Include FFMPEG support" 	ON)
option(USE_LIBJPEG 	"Include JPEG support" 		ON)

find_package(OpenCV)

if(USE_OPENCV AND OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# ========================== 3rdparty ==========================
# libjpeg
set(LIBJPEG_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")
set(LIBJPEG_LIBS "${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/lib/libjpeg.lib")

# fftw
if(USE_FFTW)
	set(FFTW_ROOT_DIR "${PROJECT_SOURCE_DIR}/3rdparty/fftw")
	set(FFTW_INCLUDE_DIRS ${FFTW_ROOT_DIR})
	set(FFTW_LIBS  "${FFTW_ROOT_DIR}/libfftw3-3.lib" "${FFTW_ROOT_DIR}/libfftw3f-3.lib" "${FFTW_ROOT_DIR}/libfftw3l-3.lib")

	include_directories(${FFTW_INCLUDE_DIRS})
endif(USE_FFTW)

# Include header files
include_directories(${LIBJPEG_INCLUDE_DIRS})

# ========================== Debug ==========================
set(CMAKE_DEBUG_POSTFIX "d")

# ========================== Modules ==========================
include_directories(${PROJECT_SOURCE_DIR}/modules)

if(USE_KINECT)
	add_subdirectory(modules/kinect)
endif(USE_KINECT)
add_subdirectory(modules/zcore)
add_subdirectory(modules/zgui)
add_subdirectory(modules/zimgproc)
add_subdirectory(modules/zfeatures2d)
add_subdirectory(modules/zmath)

enable_testing()
add_subdirectory(test)

# ========================== Install ==========================
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
	if(UNIX)
		set(CMAKE_INSTALL_PREFIX "/usr/zMatrix" CACHE PATH "Installation Directory")
	elseif(WIN32)
		set(CMAKE_INSTALL_PREFIX "D:/zMatrix" CACHE PATH "Installation Directory")
	endif()
endif()

# header files
install(FILES  ${PROJECT_SOURCE_DIR}/modules/zmatrix.h DESTINATION "${CMAKE_INSTALL_PREFIX}/include/")
if(USE_KINECT)
	install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/kinect/ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kinect" FILES_MATCHING PATTERN "*.h")
endif(USE_KINECT)
install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zcore/ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/zcore" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zgui/ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/zgui" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zimgproc/ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/zimgproc" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zfeatures2d/ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/zfeatures2d" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY  ${PROJECT_SOURCE_DIR}/modules/zmath/ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/zmath" FILES_MATCHING PATTERN "*.h")

include(cmake/CMakeUtils.cmake)

# ========================== Build platform ==========================
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