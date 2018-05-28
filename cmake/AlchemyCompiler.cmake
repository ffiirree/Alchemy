macro(add_cxx_flag flag)
    set(ALCHEMY_CXX_FLAGS "${ALCHEMY_CXX_FLAGS} ${flag}")
endmacro()

# 检查编译器对C++标准的支持
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(NOT COMPILER_SUPPORTS_CXX14)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} DO NOT SUPPORT C++14.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
    add_cxx_flag(/utf-8)
    add_cxx_flag(/wd4996)

    # W4，最高编译警告
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        add_cxx_flag(/W4)
    endif()

    # 使用内部函数的程序比较快，因为它们没有函数调用系统开销。
    add_cxx_flag(/Oi)

    # 允许编译器以打包函数(COMDATs)的形式对各个函数进行打包。
    add_cxx_flag(/Gy)
elseif(CMAKE_COMPILER_IS_GNUCXX)
    add_cxx_flag(-fsigned-char)

    add_cxx_flag(-Wextra)
    add_cxx_flag(-Wall)
    add_cxx_flag(-Werror=return-type)
    add_cxx_flag(-Werror=non-virtual-dtor)
    add_cxx_flag(-Werror=address)
    add_cxx_flag(-Werror=sequence-point)
    add_cxx_flag(-Werror=format-security -Wformat)
    add_cxx_flag(-Wmissing-declarations)
    add_cxx_flag(-Winit-self)
    add_cxx_flag(-Wpointer-arith)
    add_cxx_flag(-Wshadow)
    add_cxx_flag(-Wsign-promo)
    add_cxx_flag(-Wuninitialized)

    add_cxx_flag(-Wno-unused-parameter)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALCHEMY_CXX_FLAGS}")