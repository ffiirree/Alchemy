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
alchemy_option(USE_CUDA 	"Build Alchemy with CUDA library support"	ON)
alchemy_option(USE_CUDNN 	"Build Alchemy with CUDNN library support"	ON)
alchemy_option(USE_OPENCV 	"Build Alchemy with OpenCV library support"	ON)

alchemy_option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

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
alchemy_status("")
alchemy_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Build ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
alchemy_status("")
alchemy_status("   Name                    :" ${PROJECT_NAME})
alchemy_status("   Version                 :" ${PROJECT_VERSION})
alchemy_status("   Host                    :" ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
alchemy_status("   CMake                   :" ${CMAKE_VERSION})
alchemy_status("   CMake generator         :" ${CMAKE_GENERATOR})
alchemy_status("   CMake build tool        :" ${CMAKE_BUILD_TOOL})
if(MSVC)
	alchemy_status("   MSVC                    :" ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
	alchemy_status("   Xcode                   :" ${XCODE_VERSION})
endif()	
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
	alchemy_status("   Configuration           :" ${CMAKE_BUILD_TYPE})
endif()
alchemy_status("")

# ========================== C/C++ options ==========================
alchemy_status("")
alchemy_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ C/C++ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
alchemy_status("")
alchemy_status("   C++ 14                  :" COMPILER_SUPPORTS_CXX14 THEN YES ELSE NO)
alchemy_status("   C++ Compiler            :" ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION}))
alchemy_status("   C++ flags (Release)     :" ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
alchemy_status("   C++ flags (Debug)       :" ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
alchemy_status("   C Compiler              :" ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1})
alchemy_status("   C flags (Release)       :" ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
alchemy_status("   C flags (Debug)         :" ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
alchemy_status("")

# ========================== Dependencies ==========================
alchemy_status("")
alchemy_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dependencies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
alchemy_status("Media:")
alchemy_status("   zlib                    :" ZLIB_FOUND THEN YES ELSE NO)
alchemy_status("   libjpeg                 :" JPEG_FOUND THEN YES ELSE NO)
alchemy_status("   libpng                  :" PNG_FOUND THEN YES ELSE NO)
alchemy_status("   FFmpeg                  :" FFMPEG_FOUND THEN YES ELSE NO)
alchemy_status("Math:")
alchemy_status("   fftw                    :" FFTW_FOUND THEN YES ELSE NO)
alchemy_status("   Atlas                   :" Atlas_FOUND THEN YES ELSE NO)
alchemy_status("   NNPACK                  :" NNPACK_FOUND THEN YES ELSE NO)
alchemy_status("GPGPU:")
alchemy_status("   CUDA                    :" CUDA_FOUND THEN YES(ver ${CUDA_VERSION}) ELSE NO)
alchemy_status("   CuDNN                   :" CUDNN_FOUND THEN YES(ver ${CUDNN_VERSION}) ELSE NO)
alchemy_status("LOG:")
alchemy_status("   Glog                    :" Glog_FOUND THEN YES ELSE NO)
alchemy_status("GUI:")
if(WIN32)
    alchemy_status("   WIN32                   :" YES)
else()
    alchemy_status("   Gtk+ 2.x                :" GTK2_FOUND THEN YES(ver ${GTK2_VERSION}) ELSE NO)
endif()

# log("Install path:" ${CMAKE_INSTALL_PREFIX})
alchemy_status("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
alchemy_status("")