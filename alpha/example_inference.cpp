#include <string>
#include "inference.hpp"

int main(int argc, char* argv[]) {
    std::string path_to_unaries = "/home/rudy/datasets/MSRC/texton_unaries/2_21_s.c_unary";
    std::string path_to_output = "/home/rudy/workspace/densecrf/build/res.ppm";
    std::string path_to_image = "/home/rudy/workspace/densecrf/build/2_21_s.ppm";

    minimize_dense_alpha_divergence(path_to_image, path_to_unaries, path_to_output, 20, 15, 1);
}
