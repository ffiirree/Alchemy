# --------------------------------------------------------------
#  CMakeLists.txt for util
# --------------------------------------------------------------
file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. UTIL_SRCS)
add_library(alchemy_util STATIC ${UTIL_SRCS} ${HEADERS})

 install(TARGETS alchemy_util ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET alchemy_util PROPERTY FOLDER "modules")