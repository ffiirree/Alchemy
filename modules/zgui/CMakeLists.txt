include_directories(${PROJECT_SOURCE_DIR}/modules)
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/3rdparty/libjpeg/include")

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZGUI_SRCS)
add_library(zgui STATIC ${ZGUI_SRCS} ${HEADERS})

install(TARGETS zgui ARCHIVE DESTINATION "D:/zMatrix/lib/")

set_property(TARGET zgui PROPERTY FOLDER "modules")