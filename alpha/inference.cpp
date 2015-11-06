#include "inference.hpp"
#include "densecrf.h"
#include <iostream>
#include <string>



using namespace Eigen;




int main(int argc, char* argv[]) {

    std::string path_to_unary = "/home/rudy/datasets/MSRC/texton_unaries/2_21_s.c_unary";
    std::string path_to_output = "/home/rudy/workspace/densecrf/build/res.ppm";
    std::string path_to_image = "/home/rudy/workspace/densecrf/build/2_21_s.ppm";

    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unary, size);
    unsigned char * img = load_image(path_to_image, size);


    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));

    crf.addPairwiseBilateral( 80, 80, 13, 13, 13, img, new PottsCompatibility(10));


    MatrixXf Q = crf.inference(10);

    std::cout << "Performed inference" << std::endl;

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);



}
