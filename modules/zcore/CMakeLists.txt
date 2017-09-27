# --------------------------------------------------------------
#  CMakeLists.txt for zcore module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZGUI_SRCS)
add_library(zcore STATIC ${ZGUI_SRCS} ${HEADERS})

install(TARGETS zcore ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET zcore PROPERTY FOLDER "modules")