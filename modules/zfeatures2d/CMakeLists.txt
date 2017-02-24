# --------------------------------------------------------------
#  CMakeLists.txt for zfeatures2d module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZFEA_SRCS)
add_library(zfeatures2d STATIC ${ZFEA_SRCS} ${HEADERS})

install(TARGETS zfeatures2d ARCHIVE DESTINATION "${INSTALL_DIR}/lib/")
set_property(TARGET zfeatures2d PROPERTY FOLDER "modules")