# --------------------------------------------------------------
#  CMakeLists.txt for core module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZGUI_SRCS)
add_library(alchemy_core STATIC ${ZGUI_SRCS} ${HEADERS})

install(TARGETS alchemy_core ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET alchemy_core PROPERTY FOLDER "modules")