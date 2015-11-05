include_directories(../external/ProbImage ../external/ppm)

add_executable(inference inference.cpp)
target_link_libraries(inference probimage ppm)
