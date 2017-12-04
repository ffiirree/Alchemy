# --------------------------------------------------------------
#  CMakeLists.txt for util
# --------------------------------------------------------------
file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. UTIL_SRCS)
add_custom_target(util SOURCES ${HEADERS})
# add_library(util STATIC ${UTIL_SRCS} ${HEADERS})

# install(TARGETS util ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET util PROPERTY FOLDER "modules")