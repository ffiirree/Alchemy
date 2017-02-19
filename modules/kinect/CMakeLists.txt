include_directories(${PROJECT_SOURCE_DIR}/modules)

if (WIN32)
	# 头文件目录，OpenCV + Kinect SDK 2.0
	set(KINECT_INCLUDE "$ENV{KINECTSDK20_DIR}/inc")
	include_directories(${OPENCV_INCLUDE_DIRS})
	include_directories(${KINECT_INCLUDE})
endif()

file(GLOB_RECURSE HEADERS *.h)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZKINECT_SRCS)
add_library(zkinect STATIC ${ZKINECT_SRCS} ${HEADERS})

install(TARGETS zkinect ARCHIVE DESTINATION "D:/zMatrix/lib/")

set_property(TARGET zkinect PROPERTY FOLDER "modules")