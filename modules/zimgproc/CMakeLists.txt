include_directories(${PROJECT_SOURCE_DIR}/modules)
include_directories(${OPENCV_INCLUDE_DIRS})

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZIMGPROC_SRCS)
add_library(zimgproc STATIC ${ZIMGPROC_SRCS} ${HEADERS})

install(TARGETS zimgproc ARCHIVE DESTINATION "D:/zMatrix/lib/")

set_property(TARGET zimgproc PROPERTY FOLDER "modules")