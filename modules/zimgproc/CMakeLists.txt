# --------------------------------------------------------------
#  CMakeLists.txt for zimgproc module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZIMGPROC_SRCS)
add_library(zimgproc STATIC ${ZIMGPROC_SRCS} ${HEADERS})

install(TARGETS zimgproc ARCHIVE DESTINATION "${INSTALL_DIR}/lib/")
set_property(TARGET zimgproc PROPERTY FOLDER "modules")