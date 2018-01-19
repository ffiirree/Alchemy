# --------------------------------------------------------------
#  CMakeLists.txt for zgui module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZGUI_SRCS)
add_library(alchemy_gui STATIC ${ZGUI_SRCS} ${HEADERS})

install(TARGETS alchemy_gui ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET alchemy_gui PROPERTY FOLDER "modules")