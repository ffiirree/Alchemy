include_directories(${PROJECT_SOURCE_DIR}/modules)
include_directories(${OPENCV_INCLUDE_DIRS})

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZMATCH_SRCS)
add_library(zmatch STATIC ${ZMATCH_SRCS} ${HEADERS})

install(TARGETS zmatch ARCHIVE DESTINATION "D:/zMatrix/lib/")

set_property(TARGET zmatch PROPERTY FOLDER "modules")