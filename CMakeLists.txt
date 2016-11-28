cmake_minimum_required (VERSION 2.6)
project(densecrf)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include_directories( include )
include_directories( include/mean-shift )

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (external)
add_subdirectory (alpha)
