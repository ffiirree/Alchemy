# --------------------------------------------------------------
#  CMakeLists.txt for zgui module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZGUI_SRCS)
add_library(zgui STATIC ${ZGUI_SRCS} ${HEADERS})

install(TARGETS zgui ARCHIVE DESTINATION "${INSTALL_DIR}/lib/")
set_property(TARGET zgui PROPERTY FOLDER "modules")