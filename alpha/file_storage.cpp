#include "file_storage.hpp"
#include "ppm.h"
#include "probimage.h"
#include <iostream>


unsigned char * load_image( const std::string path_to_image, img_size size){
    int W, H;
    unsigned char * img = readPPM( path_to_image.c_str(), W, H);
    if(!img){
        std::cout << "Couldn't load the image" << std::endl;
    }
    if(size.height != H || size.width != W){
        std::cout << "Dimension doesn't correspond to unaries" << std::endl;
    }
    return img;
}


MatrixXf load_unary( const std::string path_to_unary, img_size& size) {

    ProbImage texton;
    texton.decompress(path_to_unary.c_str());
    texton.boostToProb();

    MatrixXf unaries( texton.depth(), texton.width() * texton.height());
    int i,j,k;
    for(i=0; i<texton.height(); ++i){
        for(j=0; j<texton.width(); ++j){
            for(k=0; k<texton.depth(); ++k){
                // careful with the index position, the operator takes
                // x (along width), then y (along height)

                // Also note that these are probabilities, what we
                // want are unaries, so we need to
                unaries(k, i*texton.width() + j) = -log( texton(j,i,k));
            }
        }
    }

    size = {texton.width(), texton.height()};

    return unaries;
}

void save_map(const MatrixXf estimates, const img_size size, const std::string path_to_output) {
    std::vector<short> labeling(estimates.cols());

    // MAP estimation
    for(int i=0; i<estimates.cols(); ++i) {
        int lbl;
        estimates.col(i).maxCoeff( &lbl);
        labeling[i] = lbl;
    }

    // Make the image
    unsigned char * img = new unsigned char[estimates.cols()* 3];
    for(int i=0; i<estimates.cols(); ++i) {
        unsigned char color= (labeling[i] *256*256*256 / estimates.rows());
        img[3*i + 0] = color&0xff;
        img[3*i + 1] = (color>>8)&0xff;
        img[3*i + 2] = (color>>16)&0xff;

    }

    //Write the image to the file
    writePPM(path_to_output.c_str(), size.width, size.height, img);

}
