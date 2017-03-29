#include <string>
#include "inference.hpp"
#include "file_storage.hpp"


int main(int argc, char* argv[]) {
    
    //Potts_weight_set params(3, 2, 50, 15, 30);
    //Potts_weight_set params(3, 0.5, 50, 1, 1);    // Pascal,DC-neg
    Potts_weight_set params(3.5, 2.25, 31, 8, 1.7);      // MSRC, Dc-neg
    //std::string path_to_image = "/home/tomj/Documents/4YP/densecrf/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/" + std::string(argv[1]) + ".bmp";
    //std::string path_to_unaries = "/home/tomj/Documents/4YP/densecrf/data/MSRC/texton_unaries/" + std::string(argv[1]) + ".c_unary";
    //std::string dataset_name = "MSRC";
    std::string path_to_image = "/home/tomj/Documents/4YP/densecrf/data/PascalVOC2010/JPEGImages/" + std::string(argv[1]) + ".jpg";
    std::string path_to_unaries = "/home/tomj/Documents/4YP/densecrf/data/PascalVOC2010/logit_unaries/" + std::string(argv[1]) + ".c_unary";
    std::string dataset_name = "PascalVOC2010";
    std::string path_to_output = "/home/tomj/Documents/4YP/densecrf/data/output/"  + std::string(argv[1]);
    make_dir(path_to_output);
	path_to_output = "/home/tomj/Documents/4YP/densecrf/data/output/"  + std::string(argv[1]) + "/prox_lp.bmp";
	minimize_prox_LP(path_to_image, path_to_unaries, params, path_to_output, dataset_name,argc,argv);
	double sp_constant = 0.1;
	for (int i = 0; i < 3; i++) {
		sp_constant *= 10;
		path_to_output = "/home/tomj/Documents/4YP/densecrf/data/output/"  + std::string(argv[1]) + "/" + std::to_string(sp_constant) + "_lp.bmp";
		minimize_prox_LP_super_pixels(path_to_image, path_to_unaries, params, path_to_output, dataset_name,sp_constant);
	}
    return 0;
 
}
