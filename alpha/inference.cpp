#include "inference.hpp"
#include "alpha_crf.hpp"
#include "color_to_label.hpp"
#include <ctime>
#include <iostream>
#include <string>

using namespace Eigen;

Potts_weight_set::Potts_weight_set(float spatial_std, float spatial_potts_weight,
                                   float bilat_spatial_std, float bilat_color_std,
                                   float bilat_potts_weight):spatial_std(spatial_std),
                                                             spatial_potts_weight(spatial_potts_weight),
                                                             bilat_spatial_std(bilat_spatial_std),
                                                             bilat_color_std(bilat_color_std),
                                                             bilat_potts_weight(bilat_potts_weight){}


void minimize_dense_alpha_divergence(std::string path_to_image, std::string path_to_unaries,
                                     Potts_weight_set parameters, std::string path_to_output, float alpha,
                                     std::string dataset_name) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

// Load a crf
    AlphaCRF crf(size.width, size.height, unaries.rows(), alpha);


    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    MatrixXf Q = crf.inference();

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);

}

void minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                         Potts_weight_set parameters, std::string path_to_output,
                         std::string dataset_name) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    std::cout << unaries.rows() << '\t' << unaries.cols() << '\n';

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    //crf.compute_kl_divergence();

    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.inference();
    std::cout << crf.compute_energy(Q) << '\n';
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "KL divergence: " << crf.klDivergence(Q) << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
}

void minimize_LR_QP(std::string path_to_image, std::string path_to_unaries,
                    Potts_weight_set parameters, std::string path_to_output,
                    std::string dataset_name) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    //crf.compute_kl_divergence();

    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.qp_inference();
    std::cout << crf.compute_energy(Q) << '\n';
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
}

void minimize_QP_cccp(std::string path_to_image, std::string path_to_unaries,
                      Potts_weight_set parameters, std::string path_to_output,
                      std::string dataset_name) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    //crf.compute_kl_divergence();

    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.qp_cccp_inference();
    std::cout << crf.compute_energy(Q) << '\n';
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
}



void minimize_cccp_mean_field(std::string path_to_image, std::string path_to_unaries,
                              Potts_weight_set parameters, std::string path_to_output,
                              std::string dataset_name) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));

    //crf.compute_kl_divergence();

    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.cccp_inference();
    std::cout << crf.compute_energy(Q) << '\n';
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "KL divergence: " << crf.klDivergence(Q) << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
}



void gradually_minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                   Potts_weight_set parameters, std::string path_to_output,
                                   std::string dataset_name) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));

    MatrixXf Q = crf.grad_inference();

    // Perform the MAP estimation on the fully factorized dipstribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
}

void unaries_baseline(std::string path_to_unaries, std::string path_to_output, std::string dataset_name){
    img_size size;
    MatrixXf unaries = load_unary(path_to_unaries, size);
    MatrixXf Q(unaries.rows(), unaries.cols());
    expAndNormalize(Q, -unaries);
    save_map(Q, size, path_to_output, dataset_name);
}


label_matrix minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                 Potts_weight_set parameters) {
    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    int M = unaries.rows();
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));

    MatrixXf Q = crf.inference();
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    return get_label_matrix(Q, size);
}
