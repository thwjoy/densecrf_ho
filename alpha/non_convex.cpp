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
	std::string unaries_directory = "./data/MSRC/texton_unaries/";
	std::string images_directory = "./data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/";
	//std::string unaries_directory = "./data/PascalVOC2010/logit_unaries/";
	//std::string images_directory = "./data/PascalVOC2010/JPEGImages/";
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

	Potts_weight_set params(3, 2, 50, 15, 3); //QP MSRC These seem to work the best...
	//Potts_weight_set params(1, 19.983100, 28.759019, 1.0, 39.932261);
	//Potts_weight_set params(3.071772, 0.5, 49.785678, 1, 0.960811);
	//Potts_weight_set params(11.110367,0.5,1,50,92.53338);
	//run the minimisation
	minimize_LR_QP_non_convex(path_to_image, path_to_unaries, params, path_to_output, dataset_name);
	
	return 0;

}