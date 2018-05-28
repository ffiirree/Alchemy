# --------------------------------------------------------------
#  CMakeLists.txt for math module
# --------------------------------------------------------------
file(GLOB_RECURSE HEADERS *.h *.hpp)
source_group("Header Files" FILES ${HEADERS})

file(GLOB CPP_FILES *.cpp)

if (USE_CUDA)
    file(GLOB CUDA_FILES ./*.cu)
    cuda_add_library(alchemy_math ${CPP_FILES} ${CUDA_FILES} ${HEADERS})
else()
    add_library(alchemy_math ${CPP_FILES} ${HEADERS})
endif ()

install(TARGETS alchemy_math ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
set_property(TARGET alchemy_math PROPERTY FOLDER "modules")