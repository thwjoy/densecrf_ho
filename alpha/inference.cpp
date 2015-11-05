#include "inference.hpp"
#include "probimage.h"
#include "ppm.h"
#include <iostream>
#include <string>



using namespace Eigen;




int main(int argc, char* argv[]) {

    std::string path_to_unary = "/home/rudy/datasets/MSRC/texton_unaries/2_21_s.c_unary";
    std::string path_to_output = "/home/rudy/workspace/densecrf/build/res.ppm";

    img_size size;
    MatrixXf unaries = loadUnary(path_to_unary, size);

    save_map(unaries, size, path_to_output);

}






MatrixXf loadUnary( const std::string path_to_unary, img_size& size) {

    ProbImage texton;
    texton.decompress(path_to_unary.c_str());
    texton.boostToProb();

    MatrixXf unaries( texton.width() * texton.height(), texton.depth());
    int i,j,k;
    for(i=0; i<texton.height(); ++i){
        for(j=0; j<texton.width(); ++j){
            for(k=0; k<texton.depth(); ++k){
                // careful with the index position, the operator takes x (along width), then y (along height)
                unaries(i*texton.width() + j, k) = texton(j,i,k);
            }
        }
    }

    size = {texton.width(), texton.height()};

    return unaries;
}

void save_map(const MatrixXf estimates, const img_size size, const std::string path_to_output) {
    std::vector<short> labeling(estimates.rows());

    // MAP estimation
    for(int i=0; i<estimates.rows(); ++i) {
        int lbl;
        estimates.row(i).maxCoeff( &lbl);
        labeling[i] = lbl;
    }

    // Make the image
    unsigned char * img = new unsigned char[estimates.rows()* 3];
    for(int i=0; i<estimates.rows(); ++i) {
        unsigned char color= (labeling[i] *256*256*256 / estimates.cols());
        img[3*i + 0] = color&0xff;
        img[3*i + 1] = (color>>8)&0xff;
        img[3*i + 2] = (color>>16)&0xff;

    }

    //Write the image to the file
    writePPM(path_to_output.c_str(), size.first, size.second, img);

}
