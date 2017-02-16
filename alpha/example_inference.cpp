#include <string>
#include "inference.hpp"


int main(int argc, char* argv[]) {
    
    //Potts_weight_set params(3, 2, 50, 15, 30);
    //Potts_weight_set params(3, 0.5, 50, 1, 1);    // Pascal,DC-neg
    Potts_weight_set params(3.5, 2.25, 31, 8, 1.7);      // MSRC, Dc-neg
    std::string path_to_image = "/home/tomj/Documents/4YP/densecrf/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/" + std::string(argv[1]) + ".bmp";
    std::string path_to_unaries = "/home/tomj/Documents/4YP/densecrf/data/MSRC/texton_unaries/" + std::string(argv[1]) + ".c_unary";
    std::string path_to_output = "/home/tomj/Documents/4YP/densecrf/data/output/"  + std::string(argv[1]) + "_lp.bmp";
    std::string dataset_name = "MSRC";

    minimize_prox_LP(path_to_image, path_to_unaries, params, path_to_output, dataset_name, argc, argv);

    return 0;
 
}
