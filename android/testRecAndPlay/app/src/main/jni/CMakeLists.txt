

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

find_library(log_lib log)
find_library(android_lib android)

include_directories(
     ${CMAKE_SOURCE_DIR}/base/include/
#     ${CMAKE_SOURCE_DIR}/base/include/define_variables.h
#     ${CMAKE_SOURCE_DIR}/base/include/audio_ring_buffer.h
#     ${CMAKE_SOURCE_DIR}/base/include/ring_buffer.h
#     ${CMAKE_SOURCE_DIR}/impl/include/rtns_model.h
#     ${CMAKE_SOURCE_DIR}/impl/rt_impl.h
#     ${CMAKE_SOURCE_DIR}/base/include/MNN
)

file(
     GLOB src-files
     ${CMAKE_SOURCE_DIR}/base/ring_buffer.c
     ${CMAKE_SOURCE_DIR}/base/audio_ring_buffer.cpp
     ${CMAKE_SOURCE_DIR}/base/rtns_model.cpp
     ${CMAKE_SOURCE_DIR}/impl/jni_rt_impl.cpp
     ${CMAKE_SOURCE_DIR}/impl/rt_impl.cpp
)

# mnn
#include_directories(${CMAKE_SOURCE_DIR}/MNN/include)
#include_directories(${CMAKE_SOURCE_DIR}/MNN/include/expr/)
add_library(libMNN SHARED IMPORTED)
add_library(libMNN_CL SHARED IMPORTED)
add_library(libMNN_Express SHARED IMPORTED)
set_target_properties(
        libMNN
        libMNN_CL
        libMNN_Express
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libs/MNN/${ANDROID_ABI}/libMNN.so
        ${CMAKE_SOURCE_DIR}/libs/MNN/${ANDROID_ABI}/libMNN_CL.so
        ${CMAKE_SOURCE_DIR}/libs/MNN/${ANDROID_ABI}/libMNN_Express.so
)


# 编译为动态库
add_library(
        rt_ns
        SHARED
        ${src-files})

target_link_libraries(
        rt_ns
        ${log_lib}
        ${android_lib}
        libMNN
        libMNN_CL
        libMNN_Express
)

set_target_properties(
        rt_ns
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../jnilibs/${ANDROID_ABI}"
)