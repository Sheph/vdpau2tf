cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0014 NEW)

project(VDAPP)

#platform setup

if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif ()

if (NOT TF_DIR)
    set(TF_DIR /usr/local/tensorflow CACHE STRING "tf dir" FORCE)
endif()

#out dir

set(VDAPP_OUT_DIR ${VDAPP_BINARY_DIR}/out)

#set includes/libs

set(VDAPP_INCLUDE_DIR "${VDAPP_SOURCE_DIR}/include")
include_directories(${VDAPP_INCLUDE_DIR})

#find packages

set(Boost_USE_MULTITHREADED ON)

find_package(Threads)
find_package(PkgConfig)
find_package(X11 REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(CUDA REQUIRED)

pkg_check_modules(VDPAU vdpau)

find_path(VDPAU_INCLUDE_DIR NAMES vdpau/vdpau.h vdpau/vdpau_x11.h PATHS ${VDPAU_INCLUDEDIR})
find_library(VDPAU_LIBRARY NAMES vdpau PATHS ${VDPAU_LIBDIR})
find_library(TF_LIBRARY NAMES tensorflow_cc HINTS "${TF_DIR}/bazel-bin/tensorflow" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
find_library(TF_FWK_LIBRARY NAMES tensorflow_framework HINTS "${TF_DIR}/bazel-bin/tensorflow" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

set(VDPAU_LIBRARIES ${VDPAU_LIBRARY})

#fix output directories

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VDAPP_OUT_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VDAPP_OUT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VDAPP_OUT_DIR}/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -D_REENTRANT -fPIC -DPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -D_REENTRANT -fPIC -DPIC -std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined -Wl,--exclude-libs,ALL")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined -Wl,--exclude-libs,ALL")

include_directories(${X11_INCLUDE_DIR})
include_directories(${VDPAU_INCLUDE_DIR})
include_directories(${TF_DIR})
include_directories(${TF_DIR}/bazel-genfiles)
include_directories(${TF_DIR}/bazel-tensorflow/external/eigen_archive)
include_directories(${TF_DIR}/bazel-tensorflow/external/nsync/public)
include_directories(${TF_DIR}/bazel-tensorflow/external/protobuf_archive/src)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local)

add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

add_subdirectory(vdapp)
