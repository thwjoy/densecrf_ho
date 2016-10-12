#include <string>
#include "inference.hpp"

int main(int argc, char* argv[]) {
    std::string dataset_name = "Pascal2010";
    std::string path_to_unaries;
    std::string path_to_image;
    if (dataset_name=="MSRC") {
        path_to_unaries = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/MSRC/texton_unaries/5_14_s.c_unary";
        path_to_image = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/5_14_s.bmp";
    } else if(dataset_name == "Pascal2010"){
        //path_to_unaries = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/logit_unaries/2008_000645.c_unary";
        path_to_unaries = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/logit_unaries/2007_007470.c_unary";
        //path_to_unaries = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/logit_unaries/2007_000129.c_unary";
        //path_to_image = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/JPEGImages/2008_000645.jpg";
        path_to_image = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/JPEGImages/2007_007470.jpg";
        //path_to_image = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/JPEGImages/2007_000129.jpg";
    }


    std::string path_to_alpha_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res.bmp";
    std::string path_to_mf_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-mf.bmp";
    std::string path_to_cccp_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-cccp-mf.bmp";
    std::string path_to_unoutput = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-un.bmp";
    std::string path_to_qplroutput = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-lrqp.bmp";
    std::string path_to_qpcccp_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-cccp-qp.bmp";
    std::string path_to_qpcccp_ccv_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-cccp-ccv.bmp";
    std::string path_to_fixed_iter_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-fixediter-mf.bmp";
    std::string path_to_lp_cg_line_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-lp-cg_line.bmp";
    std::string path_to_lp_sg_line_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-lp-sg_line.bmp";
    std::string path_to_prox_lp_output = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/res-prox-lp.bmp";
    // the image that we are using is from the validation set.

    //Potts_weight_set params(3, 2, 50, 15, 30);
    Potts_weight_set params(3, 0.5, 50, 1, 1);

    // minimize_dense_alpha_divergence(path_to_image, path_to_unaries, path_to_alpha_output, path_to_parameters, 5);n
    // std::cout << "Unaries" << '\n';
    // unaries_baseline(path_to_unaries, path_to_unoutput, dataset_name);
    // std::cout << "Meanfield" << '\n';
    // minimize_mean_field(path_to_image, path_to_unaries, params, path_to_mf_output, dataset_name);
    // std::cout << "CCCP Meanfield" << '\n';
    // minimize_cccp_mean_field(path_to_image, path_to_unaries, params, path_to_cccp_output, dataset_name);
    // std::cout << "Fixed Iter Meanfield"  << '\n';
    // minimize_mean_field_fixed_iter(path_to_image, path_to_unaries, params, path_to_fixed_iter_output, dataset_name, 5);
    // std::cout << "Lafferty QP"  << '\n';
    // minimize_LR_QP(path_to_image, path_to_unaries, params, path_to_qplroutput, dataset_name);
    // // std::cout << "CCCP QP" << '\n';n
    // // minimize_QP_cccp(path_to_image, path_to_unaries, params, path_to_qpcccp_output, dataset_name);
    //std::cout << "Concave QP CCCP" << '\n';
    //minimize_concave_QP_cccp(path_to_image, path_to_unaries, params, path_to_qpcccp_ccv_output, dataset_name);
    //std::cout << "LP SG line search" << '\n';
    //minimize_LP(path_to_image, path_to_unaries, params, path_to_lp_sg_line_output, dataset_name, false);
    // std::cout << "LP CG line search" << '\n';
    // minimize_LP(path_to_image, path_to_unaries, params, path_to_lp_cg_line_output, dataset_name, true);
    std::cout << "PROX LP" << '\n';
    minimize_prox_LP(path_to_image, path_to_unaries, params, path_to_prox_lp_output, dataset_name, argc, argv);
}
