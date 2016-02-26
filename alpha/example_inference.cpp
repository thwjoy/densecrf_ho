#include <string>
#include "inference.hpp"

int main(int argc, char* argv[]) {
    std::string dataset_name = "Pascal2010";
    std::string path_to_unaries;
    std::string path_to_image;
    if (dataset_name=="MSRC") {
        path_to_unaries = "/data/MSRC/texton_unaries/2_27_s.c_unary";
        path_to_image = "/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/2_27_s.bmp";
    } else if(dataset_name == "Pascal2010"){
        path_to_unaries = "/data/PascalVOC2010/unaries/2010_003547.c_unary";
        path_to_image = "/data/PascalVOC2010/JPEGImages/2010_003547.jpg";
    }


    std::string path_to_alpha_output = "/data/densecrf/res.bmp";
    std::string path_to_mf_output = "/data/densecrf/res-mf.bmp";
    std::string path_to_cccp_output = "/data/densecrf/res-cccp-mf.bmp";
    std::string path_to_unoutput = "/data/densecrf/res-un.bmp";
    std::string path_to_qplroutput = "/data/densecrf/res-lrqp.bmp";
    std::string path_to_qpcccp_output = "/data/densecrf/res-cccp-qp.bmp";
    std::string path_to_fixed_iter_output = "/data/densecrf/res-fixediter-mf.bmp";
    // the image that we are using is from the validation set.

    Potts_weight_set params(3, 3, 50, 15, 5);

    // minimize_dense_alpha_divergence(path_to_image, path_to_unaries, path_to_alpha_output, path_to_parameters, 5);n
    std::cout << "Unaries" << '\n';
    unaries_baseline(path_to_unaries, path_to_unoutput, dataset_name);
    std::cout << "Meanfield" << '\n';
    minimize_mean_field(path_to_image, path_to_unaries, params, path_to_mf_output, dataset_name);
    std::cout << "CCCP Meanfield" << '\n';
    minimize_cccp_mean_field(path_to_image, path_to_unaries, params, path_to_cccp_output, dataset_name);
    std::cout << "Lafferty QP"  << '\n';
    minimize_LR_QP(path_to_image, path_to_unaries, params, path_to_qplroutput, dataset_name);
    std::cout << "CCCP QP" << '\n';
    minimize_QP_cccp(path_to_image, path_to_unaries, params, path_to_qpcccp_output, dataset_name);
    std::cout << "Fixed Iter Meanfield"  << '\n';
    minimize_mean_field_fixed_iter(path_to_image, path_to_unaries, params, path_to_fixed_iter_output, dataset_name, 5);
}
