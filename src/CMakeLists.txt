include_directories( ../external/liblbfgs/include ../external/ )
include_directories(../alpha)
add_subdirectory(mean-shift)

add_library( densecrf util.cpp permutohedral.cpp unary.cpp pairwise.cpp objective.cpp densecrf.cpp labelcompatibility.cpp )
target_link_libraries(densecrf mean-shift qp eigen_utils)

add_library( optimization optimization.cpp )
target_link_libraries( optimization lbfgs )
