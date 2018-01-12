# --------------------------------------------------------------
#  CMakeLists.txt for zml module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})

if (USE_CUDA)
    file(GLOB CUDA_FILES *.cpp util/*.cu layers/*.cu optimizer/*.cu)
    cuda_add_library(zml STATIC ${CUDA_FILES} ${HEADERS})
else()
    file(GLOB CPP_FILES *.cpp util/*.cpp layers/*.cpp optimizer/*.cpp)
    add_library(zml STATIC ${CPP_FILES} ${HEADERS})
endif ()

install(TARGETS zml ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET zml PROPERTY FOLDER "modules")