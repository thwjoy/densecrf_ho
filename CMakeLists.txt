cmake_minimum_required (VERSION 2.6)
project(densecrf)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pg")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include_directories( include )
include_directories( include/mean-shift )

add_subdirectory (src)
add_subdirectory (external)

include_directories(external/ProbImage)

find_package(OpenCV REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Libraries
add_library(eigen_utils src/eigen_utils.cpp)

add_library(qp src/qp.cpp)

add_library(color_to_label src/color_to_label.cpp)

add_library(file_storage src/file_storage.cpp)
target_link_libraries(file_storage probimage color_to_label ${OpenCV_LIBS})

add_executable(example_inference  src/example_inference.cpp)
target_link_libraries(example_inference densecrf file_storage)

# add_subdirectory (alpha)
