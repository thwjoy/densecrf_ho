#include <string>
#include <sys/stat.h>
#include "inference.hpp"
#include "file_storage.hpp"
#include "msImageProcessor.h"
#include "libppm.h"

inline bool fileExists(const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

int main (int argc, char * argv[]) {
	std::string dataset_name = "MSRC";

	//std::string dataset_name = "PascalVOC2010";
	//std::string unaries_directory = "./data/PascalVOC2010/logit_unaries/";
	//std::string images_directory = "./data/PascalVOC2010/JPEGImages/";

	std::string unaries_directory = "/media/tomj/DATA1/4YP_data/data/MSRC/texton_unaries/";
	std::string images_directory = "/media/tomj/DATA1/4YP_data/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/";

    std::string path_to_unaries;
    std::string path_to_image;
	std::string path_to_output;
	std::string file_name;

	std::cout << "#################################################\r\nRunning non-convex segmentation\r\n#################################################\r\n";

    if (argc < 2) {
		std::cout << "Usage:  ./non_convex [file]" << std::endl;
		return 0;
	} else { 
		file_name = argv[1];
	}
	

	path_to_unaries = unaries_directory + argv[1] + std::string(".c_unary");
    path_to_image = images_directory + argv[1] + std::string(".bmp");
    path_to_output = std::string("./data/output/") + argv[1] + std::string("_out.png");

/*
    
*/
    //check the file exists
	if (!fileExists(path_to_unaries)) {
		std::cout << "Unaries not found\r\n";
		return 0;
	}
	if (!fileExists(path_to_image)) {
		std::cout << "Image not found\r\n";
	}

	std::string param1 = argv[2];
    float spc_std = std::stof(param1);
    std::string param2 = argv[3];
    float spc_potts = std::stof(param2);
    std::string param3 = argv[4];
    float bil_spcstd = std::stof(param3);
    std::string param4 = argv[5];
    float bil_colstd = std::stof(param4);
    std::string param5 = argv[6];
    float bil_potts = std::stof(param5);
    sp_params params_sp;
    if (argc==16) params_sp = sp_params {std::stof(argv[7]), std::stof(argv[8]), std::stof(argv[9]), std::stof(argv[10]), std::stof(argv[11]), std::stof(argv[12])};
    else params_sp = {0,0,0,0,0,0};

	//Potts_weight_set params(3, 2, 50, 15, 3); //QP MSRC These seem to work the best...
	//Potts_weight_set params(1, 19.983100, 28.759019, 1.0, 39.932261);// QPCCV
	Potts_weight_set params(spc_std,spc_potts,bil_spcstd,bil_colstd,bil_potts); //DC-NEG
	//Potts_weight_set params(11.110367,0.5,1,50,92.53338);// LRQP
	//Potts_weight_set params(4.105884, 77.047681, 47.793787, 4.963766, 100); //MF5

	
	//run the minimisation
	minimize_LR_QP_non_convex_tracing(path_to_image, path_to_unaries, params, params_sp, path_to_output, dataset_name, file_name);
	 
	return 0;

}
