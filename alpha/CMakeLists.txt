include_directories(../external/ProbImage ../external/ppm)

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Libraries
add_library(eigen_utils eigen_utils.cpp)

add_library(newton_cccp newton_cccp.cpp)
target_link_libraries(newton_cccp eigen_utils)

add_library(qp qp.cpp)

add_library(brute_force brute_force.cpp)

add_library(color_to_label color_to_label.cpp)

add_library(evaluation evaluation.cpp)
target_link_libraries(evaluation ${OpenCV_LIBS})

add_library(file_storage file_storage.cpp)
target_link_libraries(file_storage probimage color_to_label ${OpenCV_LIBS})

add_library(inference inference.cpp alpha_crf.cpp)
target_link_libraries(inference densecrf brute_force file_storage eigen_utils newton_cccp)

# Executables
add_executable(run throwaway.cpp)
target_link_libraries(run file_storage densecrf newton_cccp inference)

add_executable(example_inference  example_inference.cpp)
target_link_libraries(example_inference inference file_storage)

add_executable(toy-problem toy-problem.cpp)
target_link_libraries(toy-problem inference)

add_executable(evaluate evaluate_on_dataset.cpp)
target_link_libraries(evaluate inference file_storage color_to_label evaluation)

add_executable(learn_pairwise pairwise_learning.cpp)
target_link_libraries(learn_pairwise inference file_storage optimization)

add_executable(grid_search grid_search.cpp)
target_link_libraries(grid_search inference evaluation file_storage)

add_executable(cv-script cv-script.cpp)
target_link_libraries(cv-script densecrf file_storage)

add_executable(stereo stereo.cpp)
target_link_libraries(stereo densecrf file_storage inference)
