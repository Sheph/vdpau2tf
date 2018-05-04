cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0014 NEW)

project(VDAPP)

#platform setup

if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif ()

#out dir

set(VDAPP_OUT_DIR ${VDAPP_BINARY_DIR}/out)

#set includes/libs

set(VDAPP_INCLUDE_DIR "${VDAPP_SOURCE_DIR}/include")
include_directories(${VDAPP_INCLUDE_DIR})

#find packages

find_package(Threads)
find_package(PkgConfig)

#fix output directories

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VDAPP_OUT_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VDAPP_OUT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VDAPP_OUT_DIR}/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -D_REENTRANT -fPIC -DPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -D_REENTRANT -fPIC -DPIC -std=c++03")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined -Wl,--exclude-libs,ALL")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined -Wl,--exclude-libs,ALL")

add_subdirectory(vdapp)
