#include <chrono>
#include <ctime>
#include <fstream>
#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"


struct sp_params {
    float const_1;
    float norm_1;
};

void image_inference(std::string method,
                     std::string path_to_results,
                     std::string path_to_image,
                     std::string path_to_unaries,
                     float spc_std,
                     float spc_potts,
                     float bil_spcstd,
                     float bil_colstd,
                     float bil_potts,
                     LP_inf_params & lp_params,
                     sp_params params)
{


    MatrixXf Q;

    img_size size = {-1, -1};
    //Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,bil_colstd, bil_colstd, bil_colstd,img, new PottsCompatibility(bil_potts));

    Q = crf.unary_init();

    if (method == "lp_sp"){    // standard prox_lp
        std::cout << "---Running tests on proximal LP with higher order potentials\r\n";
        crf.addSuperPixel(img, 8, 4, 100, params.const_1, params.norm_1);
        try {
            Q = crf.lp_inference_prox_super_pixels(Q, lp_params);
        } catch (std::runtime_error &e) {
            std::cout << "Runtime Error!: " << e.what() << std::endl;
        }
    } else if (method == "lp") {
        std::cout << "---Running tests on proximal LP\r\n";
        Q = crf.lp_inference_prox_super_pixels(Q, lp_params);
    } else if (method == "qp"){ //qp with a non convex energy function, relaxations removed, just run without super pixels
        std::cout << "---Running tests on QP with non convex energy\r\n";
        Q = crf.qp_inference_super_pixels_non_convex(Q);
    } else if (method == "qp_sp"){ //qp with a non convex energy function, relaxations removed
        std::cout << "---Running tests on QP with non convex energy with higher order potentials\r\n";
        crf.addSuperPixel(img, 8, 4, 100, params.const_1, params.norm_1);
        Q = crf.qp_inference_super_pixels_non_convex(Q);
    } else if (method == "unary"){
        (void)0;
    } else{
        std::cout << "Unrecognised method.\n Proper error handling would do something but I won't." << '\n';
    }

    double discretized_energy;
    if (method == "qp_sp")
    {
        discretized_energy = crf.assignment_energy_higher_order(crf.currentMap(Q));
    }
    else if (method == "prox_lp_sp")
    {
        discretized_energy = crf.assignment_energy_higher_order(crf.currentMap(Q));
    }
    else
    {
        discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    }
    save_map(Q, size, path_to_results, "MSRC");
    std::cout << "# method: " << method << '\t' << discretized_energy << std::endl;
    delete[] img;


}

int main(int argc, char *argv[])
{
    if (argc<2) {
        std::cout << "Usage: ./example_inference method spc_std spc_potts bil_spcstd bil_colstd bil_potts [const_1 const_2 const_3 norm_1 norm_2 norm_3]" << '\n';
        std::cout << "Example: ./example_inference qp_sp [3 5 30 5 10 [10 1000]]" << '\n';
        return 1;
    }

    std::string method = argv[1];
    std::string image = "./data/img.bmp";
    std::string unaries = "./data/img.c_unary";
    std::string output = "./data/seg.png";

    //pairwise params
    float spc_std = 2.0;
    float spc_potts = 3.0;
    float bil_potts = 2.0;
    float bil_spcstd = 30.0;
    float bil_colstd = 8.0;


    if (argc == 7)
    {
        spc_std = std::stof(argv[2]);
        spc_potts = std::stof(argv[3]);
        bil_spcstd = std::stof(argv[4]);
        bil_colstd = std::stof(argv[5]);
        bil_potts = std::stof(argv[6]);
    }

    //params containing higher order terms
    sp_params params = {50, 1000};
    if (argc == 9)
    {
        params = sp_params {std::stof(argv[7]),
                            std::stof(argv[8])};
    }

    // lp inference params
    LP_inf_params lp_params;
    lp_params.prox_energy_tol = lp_params.dual_gap_tol;

    image_inference(method, output, image, unaries, spc_std, spc_potts,
                        bil_spcstd, bil_colstd, bil_potts, lp_params, params);

}
