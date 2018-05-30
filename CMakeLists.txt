cmake_minimum_required(VERSION 3.5)

include(cmake/CMakeUtils.cmake)
project(Alchemy VERSION 0.0.1 LANGUAGES C CXX)

# 使用项目分组，Visual studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/AlchemyCompiler.cmake)

# 设置bin和lib目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ========================== Options ==========================
xnet_option(USE_CUDA 	"Build Alchemy with CUDA library support"	ON)
xnet_option(USE_CUDNN 	"Build Alchemy with CUDNN library support"	ON)
xnet_option(USE_OPENCV 	"Build Alchemy with OpenCV library support"	ON)

xnet_option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# ========================== 3rdparty ==========================
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")
if(WIN32)
	add_custom_target(COPY_DLL_FILES ALL)
endif()

include(cmake/AlchemyDependencies.cmake)

# ========================== Debug ==========================
set(CMAKE_DEBUG_POSTFIX "d")

# ========================== Modules ==========================
include_directories(${PROJECT_SOURCE_DIR}/modules)

add_subdirectory(modules/util)
add_subdirectory(modules/math)
add_subdirectory(modules/core)
add_subdirectory(modules/imgproc)
add_subdirectory(modules/gui)
add_subdirectory(modules/nn)

list(APPEND ALCHEMY_LIBS alchemy_math alchemy_nn alchemy_gui alchemy_imgproc alchemy_util alchemy_core)

# simples
add_subdirectory(simples)

#enable_testing()
add_subdirectory(test)


# ========================== Build platform ==========================
xnet_status("")
xnet_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Build ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
xnet_status("")
xnet_status("   Name                    :" ${PROJECT_NAME})
xnet_status("   Version                 :" ${PROJECT_VERSION})
xnet_status("   Host                    :" ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
xnet_status("   CMake                   :" ${CMAKE_VERSION})
xnet_status("   CMake generator         :" ${CMAKE_GENERATOR})
xnet_status("   CMake build tool        :" ${CMAKE_BUILD_TOOL})
if(MSVC)
	xnet_status("   MSVC                    :" ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
	xnet_status("   Xcode                   :" ${XCODE_VERSION})
endif()	
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
	xnet_status("   Configuration           :" ${CMAKE_BUILD_TYPE})
endif()
xnet_status("")

# ========================== C/C++ options ==========================
xnet_status("")
xnet_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ C/C++ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
xnet_status("")
xnet_status("   C++ 14                  :" COMPILER_SUPPORTS_CXX14 THEN YES ELSE NO)
xnet_status("   C++ Compiler            :" ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION}))
xnet_status("   C++ flags (Release)     :" ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
xnet_status("   C++ flags (Debug)       :" ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
xnet_status("   C Compiler              :" ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1})
xnet_status("   C flags (Release)       :" ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
xnet_status("   C flags (Debug)         :" ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
xnet_status("")

# ========================== Dependencies ==========================
xnet_status("")
xnet_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dependencies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
xnet_status("Media:")
xnet_status("   zlib                    :" ZLIB_FOUND THEN YES ELSE NO)
xnet_status("   libjpeg                 :" JPEG_FOUND THEN YES ELSE NO)
xnet_status("   libpng                  :" PNG_FOUND THEN YES ELSE NO)
xnet_status("   FFmpeg                  :" FFMPEG_FOUND THEN YES ELSE NO)
xnet_status("Math:")
xnet_status("   fftw                    :" FFTW_FOUND THEN YES ELSE NO)
xnet_status("   Atlas                   :" Atlas_FOUND THEN YES ELSE NO)
xnet_status("   NNPACK                  :" NNPACK_FOUND THEN YES ELSE NO)
xnet_status("GPGPU:")
xnet_status("   CUDA                    :" CUDA_FOUND THEN YES(ver ${CUDA_VERSION}) ELSE NO)
xnet_status("   CuDNN                   :" CUDNN_FOUND THEN YES(ver ${CUDNN_VERSION}) ELSE NO)
xnet_status("LOG:")
xnet_status("   Glog                    :" Glog_FOUND THEN YES ELSE NO)
xnet_status("GUI:")
if(WIN32)
    xnet_status("   WIN32                   :" YES)
else()
    xnet_status("   Gtk+ 2.x                :" GTK2_FOUND THEN YES(ver ${GTK2_VERSION}) ELSE NO)
endif()

# log("Install path:" ${CMAKE_INSTALL_PREFIX})
xnet_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
xnet_status("")