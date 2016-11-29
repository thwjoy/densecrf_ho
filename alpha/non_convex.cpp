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
    std::string path_to_unaries;
    std::string path_to_image;
	std::string path_to_output;
	std::string file_name;

	std::cout << "#################################################\r\nRunning non-convex segmentation\r\n#################################################\r\n";

    if (argc < 2) {
		std::cout << "Usage:  ./non_convex [file]" << std::endl;
		file_name = "5_14_s";
	} else { 
		file_name = argv[1];
	}
	

	path_to_unaries = unaries_directory + argv[1] + std::string(".c_unary");
    path_to_image = images_directory + argv[1] + std::string(".bmp");
    path_to_output = std::string("./data/output/") + argv[1] + std::string("_out.bmp");
/*
    img_size size = {-1, -1};
    unsigned char * image = load_image(path_to_image, size);
    unsigned char * segment_image = new unsigned char[size.height * size.width * 3];
    int ** lables_out;
    float ** modes_out;
    int ** MPC_out;
	msImageProcessor m_process;
	m_process.DefineImage(image,COLOR,size.height,size.width);
	m_process.Segment(16,8,500,NO_SPEEDUP);
	m_process.GetResults(segment_image);
	int i = m_process.GetRegions(lables_out,modes_out,MPC_out);
	std::cout << *lables_out[0] << "\n";
	writePPMImage("./ouput.ppm",segment_image, size.height, size.width, 3, "");
*/
    //check the file exists
	if (!fileExists(path_to_unaries)) {
		std::cout << "Unaries not found\r\n";
		return 0;
	}
	if (!fileExists(path_to_image)) {
		std::cout << "Image not found\r\n";
	}

	Potts_weight_set params(3, 2, 50, 15, 3);
	//run the minimisation
	minimize_LR_QP_non_convex(path_to_image, path_to_unaries, params, path_to_output, dataset_name);
	
	return 0;

}