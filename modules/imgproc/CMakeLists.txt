# --------------------------------------------------------------
#  CMakeLists.txt for zimgproc module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})
aux_source_directory(. ZIMGPROC_SRCS)
add_library(alchemy_imgproc STATIC ${ZIMGPROC_SRCS} ${HEADERS})

install(TARGETS alchemy_imgproc ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET alchemy_imgproc PROPERTY FOLDER "modules")