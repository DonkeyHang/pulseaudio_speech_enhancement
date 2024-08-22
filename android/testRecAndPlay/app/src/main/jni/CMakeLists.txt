

project("realtime_ns")
cmake_minimum_required(VERSION 3.4.1)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
#====
if(NOT CMAKE_BUILD_TYPE)
     set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE -Ofast)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#=============

find_library(log-lib log)

include_directories(
     include
     source
     ${CMAKE_SOURCE_DIR}/base/define_variables.h
     ${CMAKE_SOURCE_DIR}/base/audio_ring_buffer.h
     ${CMAKE_SOURCE_DIR}/base/ring_buffer.h
     ${CMAKE_SOURCE_DIR}/impl/rt_impl.h
)

file(
     GLOB src-files
     ${CMAKE_SOURCE_DIR}/base/ring_buffer.c
     ${CMAKE_SOURCE_DIR}/base/audio_ring_buffer.cpp
     ${CMAKE_SOURCE_DIR}/impl/jni_rt_impl.cpp

)

# 编译为动态库
add_library(
        rt_ns
        SHARED
        ${src-files})

target_link_libraries(
        rt_ns
        log
)

set_target_properties(
        rt_ns
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../jnilibs/${ANDROID_ABI}"
)