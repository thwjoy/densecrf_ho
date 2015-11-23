#include "densecrf.h"
#include <Eigen/Core>
#include "file_storage.hpp"
#include <iostream>

void test_writing(){

    MatrixXf test = MatrixXf::Identity(10, 12);

    save_matrix("test.csv", test);

    MatrixXf loaded = load_matrix("test.csv");

    int diff = test.cwiseNotEqual(loaded).count();

    assert(diff == 0);

    std::cout << test  << '\n';
    std::cout  << '\n';
    std::cout << loaded << '\n';

}


void create_junk_parameters(){
    VectorXf junk_param = VectorXf::Constant(469, 1);

    save_matrix("parameters.csv", junk_param);
}



void initialize_crf_parameters(){
    img_size size;
    size.width= 320;
    size.height = 213;

    int M = 21;

    unsigned char * img = load_image("/home/rudy/datasets/MSRC/Images_by_partition/test/1_27_s.bmp", size);

    DenseCRF2D crf(size.height, size.width, M);
    // Add simple pairwise potts terms
    crf.addPairwiseGaussian( 1, 1, new MatrixCompatibility(MatrixXf::Identity(M,M)));
    // Add a longer range label compatibility term
    crf.addPairwiseBilateral(1,1,1,1,1, img, new MatrixCompatibility( 10 * MatrixXf::Identity(M,M) ));

    // The kernel parameters can't be initialized directly because teh addPairwise function pre-computes the features
    VectorXf kp(7);
    kp<< 1,1,80,80,13,13,13;
    crf.setKernelParameters(kp);


    VectorXf params(M *(M+1) + 7);

    int pos=0;
    int pairwise_size = crf.labelCompatibilityParameters().rows();
    VectorXf pair_params = crf.labelCompatibilityParameters();
    for (int i=0; i < pairwise_size; i++) {
        params(pos+i) = pair_params(i);
    }
    pos += pairwise_size;

    int kernel_size = crf.kernelParameters().rows();
    VectorXf kernel_params = crf.kernelParameters();
    for (int i=0; i < kernel_size; i++) {
        params(pos+i) = kernel_params(i);
    }

    save_matrix("parameters.csv", params);


}


int main(int argc, char *argv[])
{
    initialize_crf_parameters();
    return 0;
}