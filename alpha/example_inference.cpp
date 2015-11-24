#include <string>
#include "inference.hpp"

int main(int argc, char* argv[]) {
    std::string path_to_unaries = "/home/rudy/datasets/MSRC/texton_unaries/2_21_s.c_unary";
    std::string path_to_output = "/home/rudy/workspace/densecrf/build/res.bmp";
    std::string path_to_mf_output = "/home/rudy/workspace/densecrf/build/res-mf.bmp";
    std::string path_to_image = "/home/rudy/datasets/MSRC/MSRC_ObjCategImageDatabase_v2/Images/2_21_s.bmp";

    std::string path_to_parameters = "/home/rudy/workspace/densecrf/clang/alpha/latest_parameters.csv";

    minimize_dense_alpha_divergence(path_to_image, path_to_unaries, path_to_output, path_to_parameters, 20);

    minimize_mean_field(path_to_image, path_to_unaries,  path_to_mf_output, path_to_parameters);
}
