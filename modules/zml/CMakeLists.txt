# --------------------------------------------------------------
#  CMakeLists.txt for zml module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZML_SRCS)
add_library(zml STATIC ${ZML_SRCS} ${HEADERS})

install(TARGETS zml ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET zml PROPERTY FOLDER "modules")