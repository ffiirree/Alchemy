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

# 使用项目分组，Visual studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# 检查编译器对C++标准的支持
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(NOT COMPILER_SUPPORTS_CXX14)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} DO NOT SUPPORT C++14.")
endif()

# 编译要求，使用C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
	# 源代码为utf-8无BOM编译错误
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

	# 阻止warning C4996，printf->printf_s的警告
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")

	# W4，最高编译警告
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
	endif()

	# 使用内部函数的程序比较快，因为它们没有函数调用系统开销。
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi")

	# 允许编译器以打包函数(COMDATs)的形式对各个函数进行打包。
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Gy")
elseif(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")

    # enable extra warning
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    # warn all, 所有警告
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type -Werror=non-virtual-dtor \
        -Werror=address -Werror=sequence-point -Wformat -Werror=format-security \
        -Wmissing-declarations -Wundef -Wuninitialized -Winit-self")
endif()

# 设置bin和lib目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ========================== Options ==========================
option(USE_KINECT 	"Include Kinect support"	OFF)

# ========================== 3rdparty ==========================
# fftw
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")
if(WIN32)
	add_custom_target(COPY_DLL_FILES ALL)
endif()

include(cmake/FindGUI.cmake)
include(cmake/FindImage.cmake)
include(cmake/FindLibsVideo.cmake)
include(cmake/FindOCV.cmake)

find_package(Boost REQUIRED)
if(Boost_FOUND)
	set(USE_BOOST ON)
	include_directories(${Boost_INCLUDE_DIRS})
endif()

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
add_subdirectory(modules/util)
add_subdirectory(modules/zgui)
add_subdirectory(modules/zimgproc)
add_subdirectory(modules/zfeatures2d)
add_subdirectory(modules/zmath)
add_subdirectory(modules/zml)

enable_testing()
add_subdirectory(test)

# ========================== Install ==========================
# include(cmake/Install.cmake)


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
log("")

# ========================== C/C++ options ==========================
log("")
log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ C/C++ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log("")
if(NOT BUILD_SHARED_LIBS)
	log("    Built as dynamic libs?:" NO)
elseif()
	log("    Built as dynamic libs?:" YES)
endif()
if(COMPILER_SUPPORTS_CXX11)
	log("    C++11:" YES)
else()
	log("    C++11:" NO)
endif()
if(COMPILER_SUPPORTS_CXX11)
	log("    C++14:" YES)
else()
	log("    C++14:" NO)
endif()
if(USE_BOOST)
	log("    Boost:"				  ${Boost_LIB_VERSION})
endif()
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
log("")


# ========================== Media ==========================
log("")
log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Media ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log("")
if(USE_JPEG)
    log("    JPEG:" YES)
else()
    log("    JPEG:" NO)
endif()

if(USE_FFTW)
    log("    FFTW:" YES)
else()
    log("    FFTW:" NO)
endif()

if(USE_FFMPEG)
    log("    FFMPEG:" YES)
else()
    log("    FFMPEG:" NO)
endif()
log("")


# ========================== GUI ==========================
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

# log("Install path:" ${CMAKE_INSTALL_PREFIX})
log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log("")