include_directories(../external/ProbImage ../external/ppm ../external/ibfs ../external/ )

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Libraries
add_library(eigen_utils eigen_utils.cpp)

#add_library(runibfs runibfs.cpp)
#target_link_libraries(runibfs ibfs)

add_library(qp qp.cpp)

add_library(color_to_label color_to_label.cpp)

add_library(file_storage file_storage.cpp)
target_link_libraries(file_storage probimage color_to_label ${OpenCV_LIBS})

add_executable(example_inference  example_inference.cpp)
target_link_libraries(example_inference densecrf file_storage)



