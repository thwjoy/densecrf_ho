#include <string>
#include "inference.hpp"

int main(int argc, char* argv[]) {
    std::string path_to_unaries = "/home/rudy/datasets/MSRC/texton_unaries/2_27_s.c_unary";
    std::string path_to_alpha_output = "/home/rudy/workspace/densecrf/build/res.bmp";
    std::string path_to_mf_output = "/home/rudy/workspace/densecrf/build/res-mf.bmp";
    std::string path_to_cccp_output = "/home/rudy/workspace/densecrf/build/res-cccp-mf.bmp";
    std::string path_to_unoutput = "/home/rudy/workspace/densecrf/build/res-un.bmp";
    // the image that we are using is from the validation set.
    std::string path_to_image = "/home/rudy/datasets/MSRC/MSRC_ObjCategImageDatabase_v2/Images/2_27_s.bmp";

    std::string path_to_parameters = "/home/rudy/workspace/densecrf/build/alpha/learned_parameters.csv";



    //    minimize_dense_alpha_divergence(path_to_image, path_to_unaries, path_to_alpha_output, path_to_parameters, 5);
    //    unaries_baseline(path_to_unaries, path_to_unoutput);
    std::cout << "Standard Meanfield" << '\n';
    minimize_mean_field(path_to_image, path_to_unaries,  path_to_mf_output, path_to_parameters);
    std::cout << "CCCP Meanfield" << '\n';
    minimize_cccp_mean_field(path_to_image, path_to_unaries, path_to_cccp_output, path_to_parameters);
}
