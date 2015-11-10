#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "inference.hpp"

std::string stringreplace(std::string s,
                          std::string toReplace,
                          std::string replaceWith)
{
  if (s.find(toReplace) != std::string::npos){
    return(s.replace(s.find(toReplace), toReplace.length(), replaceWith));
  }

  return s;
}


std::vector<std::string> get_all_test_files(std::string path_to_dataset){
    std::string path_to_testlist = path_to_dataset + "split/Test.txt";

    std::vector<std::string> test_images;
    std::string next_img_name;
    std::ifstream file(path_to_testlist.c_str());

    while(getline(file, next_img_name)){
        test_images.push_back(next_img_name);
    }

    return test_images;
}

void do_inference(std::string path_to_images, std::string path_to_unaries,
                  std::string path_to_results, std::string image_name) {
    std::string image_path = path_to_images + image_name;

    std::string unaries_path = path_to_unaries + image_name;
    unaries_path = stringreplace(path_to_unaries, ".bmp", ".c_unary");

    std::string output_path = path_to_results + image_name;
    output_path = stringreplace(output_path, ".bmp", "_res.bmp");

    minimize_dense_alpha_divergence(image_path, unaries_path, output_path,
                                    0.5, 3, 10);
}


int main(int argc, char *argv[])
{
    if (argc<3) {
        std::cout << "evaluate path_to_dataset path_to_results" << '\n';
        return 1;
    }

    std::string path_to_dataset = argv[1];
    std::string path_to_results = argv[2];

    std::vector<std::string> test_images = get_all_test_files(path_to_dataset);
    std::string path_to_images = path_to_dataset + "MSCR_ObjCategImageDatabase_v2/Images/";
    std::string path_to_unaries = path_to_dataset + "texton_unaries/";

    for (std::vector<std::string>::iterator image_name = test_images.begin(); image_name != test_images.end(); ++image_name) {
        do_inference(path_to_images, path_to_unaries, path_to_results, *image_name);
    }

    return 0;
}
