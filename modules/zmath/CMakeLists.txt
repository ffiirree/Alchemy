# --------------------------------------------------------------
#  CMakeLists.txt for zmath module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZMATH_SRCS)
add_library(zmath STATIC ${ZMATH_SRCS} ${HEADERS})

install(TARGETS zmath ARCHIVE DESTINATION "${INSTALL_DIR}/lib/")
set_property(TARGET zmath PROPERTY FOLDER "modules")