# --------------------------------------------------------------
#  Root CMake file for zMatrix
#
#     From the off-tree build directory, invoke:
#       $ cmake <PATH_TO_ZMATRIX_ROOT>
# --------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)

include(cmake/CMakeUtils.cmake)

set(PROJECT_NAME zMatrix)
project(${PROJECT_NAME})

# 使用项目分组
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# 编译要求，使用C++11
set(CMAKE_CXX_STANDARD 14)
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ========================== Options ==========================
option(USE_OPENCV 	"Include OpenCV support" 	OFF)
option(USE_KINECT 	"Include Kinect support"	OFF)
option(USE_FFTW		"Include FFTW support"		ON)
option(USE_LIBJPEG 	"Include JPEG support" 		ON)

# ========================== 3rdparty ==========================
# fftw
if(WIN32)
	set(FFTW_ROOT_DIR "${PROJECT_SOURCE_DIR}/3rdparty/fftw")
	set(FFTW_INCLUDE_DIRS ${FFTW_ROOT_DIR})
	set(FFTW_LIBS  "${FFTW_ROOT_DIR}/libfftw3-3.lib" "${FFTW_ROOT_DIR}/libfftw3f-3.lib" "${FFTW_ROOT_DIR}/libfftw3l-3.lib")

	include_directories(${FFTW_INCLUDE_DIRS})
else()
    set(USE_FFTW OFF)
endif()

include(cmake/FindGUI.cmake)
include(cmake/FindImage.cmake)
include(cmake/FindOCV.cmake)

# ========================== configuration ==========================
include(cmake/ConfigHeaders.cmake)

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
include(cmake/Install.cmake)


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



# ========================== Image ==========================
log("")
log("")
log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log("")
if(USE_JPEG)
    log("    JPEG:" YES)
else()
    log("    JPEG:" NO)
endif()

if(USE_PNG)
    log("    PNG:" YES)
else()
    log("    PNG:" NO)
endif()
log("")
log("")


# ========================== GUI ==========================
log("")
log("")
log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GUI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log("")
if(USE_WIN32)
    log("    WIN32:" YES)
else()
    log("    WIN32:" NO)
endif()

if(USE_GTK2)
    log("    Gtk+ 2.x:" YES)
else()
    log("    Gtk+ 2.x:" NO)
endif()
log("")
log("")