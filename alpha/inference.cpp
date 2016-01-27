#include "inference.hpp"
#include "alpha_crf.hpp"
#include "color_to_label.hpp"
#include <ctime>
#include <iostream>
#include <string>

using namespace Eigen;

void minimize_dense_alpha_divergence(std::string path_to_image, std::string path_to_unaries,
                                     std::string path_to_output, std::string path_to_parameters, float alpha) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);
    VectorXf pairwise_parameters = load_matrix(path_to_parameters);

    // Load a crf
    AlphaCRF crf(size.width, size.height, unaries.rows(), alpha);


    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3,3, new PottsCompatibility(3));
    crf.addPairwiseBilateral( 40,40,2.5,2.5,2.5, img, new PottsCompatibility(3.5));
    MatrixXf Q = crf.inference();

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);

}

void minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                         std::string path_to_output, std::string path_to_parameters) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);
    VectorXf pairwise_parameters = load_matrix(path_to_parameters);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3,3, new PottsCompatibility(3));
    crf.addPairwiseBilateral(40,40,2.5,2.5,2.5, img, new PottsCompatibility(3.5));
    //crf.compute_kl_divergence();

    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.inference();
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "KL divergence: " << crf.klDivergence(Q) << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);
}

void minimize_cccp_mean_field(std::string path_to_image, std::string path_to_unaries,
                              std::string path_to_output, std::string path_to_parameters) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);
    VectorXf pairwise_parameters = load_matrix(path_to_parameters);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3,3, new PottsCompatibility(3));
    crf.addPairwiseBilateral(40,40,2.5,2.5,2.5, img, new PottsCompatibility(3.5));

    //crf.compute_kl_divergence();

    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.cccp_inference();
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "KL divergence: " << crf.klDivergence(Q) << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);
}



void gradually_minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                   std::string path_to_output, std::string path_to_parameters) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);
    VectorXf pairwise_parameters = load_matrix(path_to_parameters);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3,3,new PottsCompatibility(3));
    crf.addPairwiseBilateral(40,40,2.5,2.5,2.5, img, new PottsCompatibility(3.5));

    MatrixXf Q = crf.grad_inference();

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output);
}

void unaries_baseline(std::string path_to_unaries, std::string path_to_output){
    img_size size;
    MatrixXf unaries = load_unary(path_to_unaries, size);
    MatrixXf Q(unaries.rows(), unaries.cols());
    expAndNormalize(Q, -unaries);
    save_map(Q, size, path_to_output);
}


label_matrix minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                 float w1, float sigma_alpha, float sigma_beta) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(3,3, new PottsCompatibility(3));
    crf.addPairwiseBilateral(sigma_alpha, sigma_alpha, sigma_beta, sigma_beta, sigma_beta, img, new PottsCompatibility(w1));

    MatrixXf Q = crf.inference();
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    return get_label_matrix(Q, size);
}
