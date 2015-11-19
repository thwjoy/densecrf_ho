#include "inference.hpp"
#include "alpha_crf.hpp"
#include <iostream>
#include <string>

using namespace Eigen;

void minimize_dense_alpha_divergence(std::string path_to_image, std::string path_to_unaries, std::string path_to_output, float alpha) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    AlphaCRF crf(size.width, size.height, unaries.rows(), alpha);

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));

    crf.addPairwiseBilateral( 80, 80, 13, 13, 13, img, new PottsCompatibility(10));
    MatrixXf Q = crf.inference();

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);

}

void minimize_mean_field(std::string path_to_image, std::string path_to_unaries, std::string path_to_output) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));

    crf.addPairwiseBilateral( 80, 80, 13, 13, 13, img, new PottsCompatibility(10));
    MatrixXf Q = crf.inference();

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);
}

void gradually_minimize_mean_field(std::string path_to_image, std::string path_to_unaries, std::string path_to_output) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));

    crf.addPairwiseBilateral( 80, 80, 13, 13, 13, img, new PottsCompatibility(10));
    MatrixXf Q = crf.grad_inference();

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);
}

void unaries_baseline(std::string path_to_unaries, std::string path_to_output){
    img_size size;
    MatrixXf unaries = load_unary(path_to_unaries, size);
    save_map(unaries, size, path_to_output);
}
