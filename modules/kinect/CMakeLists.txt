# --------------------------------------------------------------
#  CMakeLists.txt for Kinect module
# --------------------------------------------------------------
if (WIN32)
	# 头文件目录Kinect SDK 2.0
	include_directories("$ENV{KINECTSDK20_DIR}/inc")
endif()

file(GLOB_RECURSE HEADERS *.h)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZKINECT_SRCS)
add_library(zkinect STATIC ${ZKINECT_SRCS} ${HEADERS})

install(TARGETS zkinect ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET zkinect PROPERTY FOLDER "modules")