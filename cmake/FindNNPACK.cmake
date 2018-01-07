# - Try to find NNPACK
#
# The following variables are optionally searched for defaults
#  NNPACK_ROOT_DIR:            Base directory where all NNPACK components are found
#
# The following are set after configuration is done:
#  NNPACK_FOUND
#  NNPACK_INCLUDE_DIRS
#  NNPACK_LIBRARIES
#  NNPACK_LIBRARYRARY_DIRS


set(NNPACK_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/include/nnpack
        $ENV{NNPACK_ROOT_DIR}
        $ENV{NNPACK_ROOT_DIR}/include
        )

set(NNPACK_LIB_SEARCH_PATHS
        /usr/lib
        $ENV{NNPACK_ROOT_DIR}
        $ENV{NNPACK_ROOT_DIR}/lib
        )

find_path(NNPACK_INCLUDE_DIR   NAMES nnpack.h
        PATHS ${NNPACK_INCLUDE_SEARCH_PATHS})

find_path(PTHREADPOOL_INCLUDE_DIR  NAMES pthreadpool.h
        PATHS ${NNPACK_INCLUDE_SEARCH_PATHS})

find_library(NNPACK_LIBRARY NAMES  nnpack
        PATHS ${NNPACK_LIB_SEARCH_PATHS})

find_library(CPUINFO_LIBRARY NAMES cpuinfo
        PATHS ${NNPACK_LIB_SEARCH_PATHS})

find_library(PTHREADPOOL_LIBRARY NAMES   pthreadpool
        PATHS ${NNPACK_LIB_SEARCH_PATHS})

set(LOOKED_FOR
        NNPACK_INCLUDE_DIR
        PTHREADPOOL_INCLUDE_DIR

        NNPACK_LIBRARY
        CPUINFO_LIBRARY
        PTHREADPOOL_LIBRARY
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NNPACK DEFAULT_MSG ${LOOKED_FOR})

if(NNPACK_FOUND)
  set(NNPACK_INCLUDE_DIRS ${NNPACK_INCLUDE_DIR} ${PTHREADPOOL_INCLUDE_DIR})
  set(NNPACK_LIBRARIES ${NNPACK_LIBRARY} ${PTHREADPOOL_LIBRARY} ${CPUINFO_LIBRARY})
  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found NNPACK      (include: ${NNPACK_INCLUDE_DIRS}, library: ${NNPACK_LIBRARY})")
  message(STATUS "Found PTHREADPOOL (include: ${PTHREADPOOL_INCLUDE_DIR}, library: ${PTHREADPOOL_LIBRARY})")
  message(STATUS "Found CPUINFO     (library: ${CPUINFO_LIBRARY})")
endif(NNPACK_FOUND)

