include_directories(../external/ProbImage)

add_executable(inference inference.cpp)
target_link_libraries(inference probimage)
