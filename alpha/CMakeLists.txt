include_directories(../external/ProbImage ../external/ppm)

find_package(OpenCV REQUIRED)

add_library(inference inference.cpp file_storage.cpp alpha_crf.cpp)
target_link_libraries(inference densecrf probimage ${OpenCV_LIBS})

add_library(brute_force brute_force.cpp)

add_executable(example_inference  example_inference.cpp)
target_link_libraries(example_inference inference)

add_executable(toy-problem toy-problem.cpp)
target_link_libraries(toy-problem brute_force inference)

add_executable(evaluate evaluate_on_dataset.cpp)
target_link_libraries(evaluate inference)

add_custom_target(example
  ./example_inference
  DEPENDS example_inference)

add_custom_target(seg
  rm -rf pred/ && mkdir pred && ./evaluate /home/rudy/datasets/MSRC/ ./pred/
  DEPENDS evaluate)
