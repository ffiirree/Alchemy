# --------------------------------------------------------------
#  CMakeLists.txt for zmatch module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZMATCH_SRCS)
add_library(zmatch STATIC ${ZMATCH_SRCS} ${HEADERS})

install(TARGETS zmatch ARCHIVE DESTINATION "${INSTALL_DIR}/lib/")
set_property(TARGET zmatch PROPERTY FOLDER "modules")