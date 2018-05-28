# --------------------------------------------------------------
#  CMakeLists.txt for nn module
# --------------------------------------------------------------

file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})

file(GLOB CPP_FILES *.cpp layers/*.cpp)

if (USE_CUDA)
    file(GLOB CUDA_FILES layers/*.cu)
    cuda_add_library(alchemy_nn ${CPP_FILES} ${CUDA_FILES} ${HEADERS})
else()
    add_library(alchemy_nn ${CPP_FILES} ${HEADERS})
endif ()

install(TARGETS alchemy_nn ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET alchemy_nn PROPERTY FOLDER "modules")