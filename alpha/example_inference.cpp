#include <string>
#include "inference.hpp"

int main(int argc, char* argv[]) {
    std::string path_to_unaries = "/data/MSRC/texton_unaries/2_27_s.c_unary";
    std::string path_to_alpha_output = "/data/densecrf/res.bmp";
    std::string path_to_mf_output = "/data/densecrf/res-mf.bmp";
    std::string path_to_cccp_output = "/data/densecrf/res-cccp-mf.bmp";
    std::string path_to_unoutput = "/data/densecrf/res-un.bmp";
    std::string path_to_qplroutput = "/data/densecrf/res-lrqp.bmp";
    std::string path_to_qpcccp_output = "/data/densecrf/res-cccp-qp.bmp";
    // the image that we are using is from the validation set.
    std::string path_to_image = "/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/2_27_s.bmp";

    std::string path_to_parameters = "/data/densecrf/alpha/learned_parameters.csv";



    // minimize_dense_alpha_divergence(path_to_image, path_to_unaries, path_to_alpha_output, path_to_parameters, 5);n
    // unaries_baseline(path_to_unaries, path_to_unoutput);
    // minimize_mean_field(path_to_image, path_to_unaries,  path_to_mf_output, path_to_parameters);
    // minimize_cccp_mean_field(path_to_image, path_to_unaries, path_to_cccp_output, path_to_parameters);
    // minimize_LR_QP(path_to_image, path_to_unaries, path_to_qplroutput, path_to_parameters);
    minimize_QP_cccp(path_to_image, path_to_unaries, path_to_qpcccp_output, path_to_parameters);
}
