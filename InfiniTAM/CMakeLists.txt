cmake_minimum_required(VERSION 3.1)
project(InfiniTAM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

macro(CANONIFY_BOOL var)
  if (${var})
    set(${var} TRUE)
  else()
    set(${var} FALSE)
  endif()
endmacro()


find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PCL REQUIRED)

find_package(CUDA QUIET)
CANONIFY_BOOL(CUDA_FOUND)
CANONIFY_BOOL(PCL_FOUND)
##################################
#message("this is CUDA_FOUND----------------" ${CUDA_FOUND})
#set(CUDA_FOUND FALSE)
#message("this is CUDA_FOUND AFTER MODIFIED----------------" ${CUDA_FOUND})
##################################
message(STATUS "CUDA found: " ${CUDA_FOUND})
message(STATUS "PCL found: " ${PCL_FOUND})

#find_package(OpenNI QUIET)
find_package(OpenNI QUIET)
CANONIFY_BOOL(OPENNI_FOUND)
message(STATUS "OpenNI found: " ${OPENNI_FOUND})

find_package(LibRoyale QUIET)
CANONIFY_BOOL(LibRoyale_FOUND)
message(STATUS "LibRoyale found: " ${LibRoyale_FOUND})

find_package(libuvc QUIET)
CANONIFY_BOOL(libuvc_FOUND)
message(STATUS "libuvc found: ${libuvc_FOUND}")

find_package(RealSense QUIET)
CANONIFY_BOOL(REALSENSE_FOUND)
message(STATUS "librealsense found: ${REALSENSE_FOUND}")

find_package(PNG QUIET)
CANONIFY_BOOL(PNG_FOUND)
message(STATUS "libpng found: ${PNG_FOUND}")

OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})
message("this is WITH_CUDA---------" ${WITH_CUDA})

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenMP.cmake)


OPTION(WITH_REALSENSE "Build with Intel RealSense support?" ${REALSENSE_FOUND})
OPTION(WITH_LIBROYALE "Build with LibRoyale support?" ${LibRoyale_FOUND})

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  #set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")  #这个是默认的
  set(CMAKE_CXX_FLAGS "-fPIC -O0 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
  #set(CMAKE_CXX_FLAGS "-fPIC -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
ENDIF()

#################################################
include_directories("/usr/local/include/pcl-1.8")
include_directories("/usr/include/eigen3")
message("PCL_INCLUDE:" ${PCL_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#################################################


add_subdirectory(ORUtils)
add_subdirectory(ITMLib)
add_subdirectory(Utils)
add_subdirectory(Engine)

########################################
IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()
#add by myself
#include_directories(${CUDA_INCLUDE_DIRS})
#add_definitions(-DCOMPILE_WITHOUT_CUDA)
#######################################

#set(PCL_LIB /usr/local/lib/libpcl_common.so /usr/local/lib/libpcl_io.so)

#add_executable(InfiniTAM_cli InfiniTAM_cli.cpp)
#target_link_libraries(InfiniTAM_cli Engine)
#target_link_libraries(InfiniTAM_cli Utils)
add_executable(InfiniTAM InfiniTAM.cpp)
target_link_libraries(InfiniTAM Engine)
target_link_libraries(InfiniTAM ${PCL_LIBRARIES})
target_link_libraries(InfiniTAM Utils)

