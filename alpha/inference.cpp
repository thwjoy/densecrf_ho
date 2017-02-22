#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "alpha_crf.hpp"
#include "qp.hpp"
#include "color_to_label.hpp"
#include "inference.hpp"

using namespace Eigen;

#define DEFAULT_SIZE -1

Potts_weight_set::Potts_weight_set(float spatial_std, float spatial_potts_weight,
                                   float bilat_spatial_std, float bilat_color_std,
                                   float bilat_potts_weight):spatial_std(spatial_std),
                                                             spatial_potts_weight(spatial_potts_weight),
                                                             bilat_spatial_std(bilat_spatial_std),
                                                             bilat_color_std(bilat_color_std),
                                                             bilat_potts_weight(bilat_potts_weight){}

void write_down_perf(double timing, double final_energy, double rounded_energy, std::string path_to_output){
    std::string txt_output = path_to_output;
    txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");

    std::ofstream txt_file(txt_output.c_str());
    txt_file << timing << '\t' << final_energy << '\t' << rounded_energy << std::endl;
    txt_file.close();
}

void minimize_dense_alpha_divergence(std::string path_to_image, std::string path_to_unaries,
                                     Potts_weight_set parameters, std::string path_to_output, float alpha,
                                     std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    AlphaCRF crf(size.width, size.height, unaries.rows(), alpha);


    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.inference();
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;

}

void minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                         Potts_weight_set parameters, std::string path_to_output,
                         std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    crf.compute_kl_divergence();
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);

    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "KL divergence: " << crf.klDivergence(Q) << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;
}

void minimize_mean_field_fixed_iter(std::string path_to_image, std::string path_to_unaries,
                                    Potts_weight_set parameters, std::string path_to_output,
                                    std::string dataset_name, int num_iter) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    crf.compute_kl_divergence();
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.inference(init, num_iter);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);

    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "KL divergence: " << crf.klDivergence(Q) << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;
}



void minimize_LR_QP(std::string path_to_image, std::string path_to_unaries,
                    Potts_weight_set parameters, std::string path_to_output,
                    std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    //crf.compute_kl_divergence();
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.qp_inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);

    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;
}

void minimize_LR_QP_non_convex(std::string path_to_image, std::string path_to_unaries,
                    Potts_weight_set parameters, std::string path_to_output,
                    std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    std::cout << "---Running mean-shift and adding super pixel term" <<std::endl;
    crf.addSuperPixel(img,4,2,2000);
    crf.addSuperPixel(img,4,2,500);
    crf.addSuperPixel(img,4,2,100);
    MatrixXf init = crf.unary_init();
    std::vector<perf_measure> traced_perfs_qp;
    std::vector<perf_measure> traced_perfs_qp_nc;
    std::vector<perf_measure> traced_perfs_qp_sp;
 /*   
    //run the inference with the convex problem
    std::cout << "---Finding global optimum, of convex energy function" <<std::endl;
    clock_t start, end;
    start = clock();
    const MatrixXf Q = crf.qp_inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
/*
    //run with non_convex function
    std::cout << "---Finding local optimum, of non-convex energy function" <<std::endl;
    path_to_output.replace(path_to_output.end()-4, path_to_output.end(),"_nc.bmp");
    clock_t start_nc, end_nc;
    start_nc = clock();
    //MatrixXf Q_non_convex = Q;
    //(void) crf.tracing_qp_inference_super_pixels_non_convex(Q_non_convex);
    MatrixXf Q_non_convex = crf.qp_inference_non_convex(init);
    end_nc = clock();
    double timing_non_convex = (double(end_nc-start_nc)/CLOCKS_PER_SEC);
    double final_energy_non_convex = crf.compute_energy(Q_non_convex);
    double discretized_energy_non_convex = crf.assignment_energy(crf.currentMap(Q_non_convex));
    write_down_perf(timing_non_convex, final_energy_non_convex, discretized_energy_non_convex, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q_non_convex, size, path_to_output, dataset_name);
/*

    //we now need to run the code with a non_convex energy function including the super pixels
    std::cout << "---Finding local optimum with super pixel" <<std::endl;
    path_to_output.replace(path_to_output.end()-7, path_to_output.end(),"_sp.bmp");
    clock_t start_sp, end_sp;
    start_sp = clock();
    //MatrixXf Q_non_convex_sp = Q;
    //(void) crf.tracing_qp_inference_super_pixels_non_convex(Q_non_convex_sp);
    MatrixXf Q_sp = crf.qp_inference_super_pixels(Q);
    end_sp = clock();
    double timing_sp = (double(end_sp-start_sp)/CLOCKS_PER_SEC);
    double final_energy_sp = crf.compute_energy(Q_sp);
    double discretized_energy_sp = crf.assignment_energy(crf.currentMap(init));
    write_down_perf(timing_sp, final_energy_sp, discretized_energy_sp, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q_sp, size, path_to_output, dataset_name);
*/
    /*
    //we now need to run the code with a non_convex energy function including the super pixels
    std::cout << "---Finding local optimum, of non-convex energy function with super pixel" <<std::endl;
    path_to_output.replace(path_to_output.end()-7, path_to_output.end(),"_nc_sp.bmp");
    clock_t start_nc_sp, end_nc_sp;
    start_nc_sp = clock();
    //MatrixXf Q_non_convex_sp = Q;
    //(void) crf.tracing_qp_inference_super_pixels_non_convex(Q_non_convex_sp);
    MatrixXf Q_non_convex_sp = crf.qp_inference_super_pixels_non_convex(Q);
    end_nc_sp = clock();
    double timing_non_convex_sp = (double(end_nc_sp-start_nc_sp)/CLOCKS_PER_SEC);
    double final_energy_non_convex_sp = crf.compute_energy(Q_non_convex_sp);
    double discretized_energy_non_convex_sp = crf.assignment_energy(crf.currentMap(Q_non_convex_sp));
    write_down_perf(timing_non_convex_sp, final_energy_non_convex_sp, discretized_energy_non_convex_sp, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q_non_convex_sp, size, path_to_output, dataset_name);
*/
    //we now need to run the code with a non_convex energy function including the super pixels starting at the initial value
    std::cout << "---Finding local optimum, of non-convex energy function with super pixel from initial values" <<std::endl;
    path_to_output.replace(path_to_output.end()-4, path_to_output.end(),"_g_nc_sp.png");
    clock_t start_g_nc_sp, end_g_nc_sp;
    start_g_nc_sp = clock();
    //MatrixXf Q_non_convex_sp = Q;
    //(void) crf.tracing_qp_inference_super_pixels_non_convex(Q_non_convex_sp);
    MatrixXf Q_g_non_convex_sp = crf.qp_inference_super_pixels_non_convex(init);
    end_g_nc_sp = clock();
    double timing_g_non_convex_sp = (double(end_g_nc_sp-start_g_nc_sp)/CLOCKS_PER_SEC);
    std::cout << "Time: " << timing_g_non_convex_sp << std::endl;    
    double final_energy_g_non_convex_sp = crf.compute_energy(Q_g_non_convex_sp);
    double discretized_energy_g_non_convex_sp = crf.assignment_energy(crf.currentMap(Q_g_non_convex_sp));
    write_down_perf(timing_g_non_convex_sp, final_energy_g_non_convex_sp, discretized_energy_g_non_convex_sp, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q_g_non_convex_sp, size, path_to_output, dataset_name);

    delete[] img;
}

void minimize_LR_QP_non_convex_tracing(std::string path_to_image, std::string path_to_unaries,
                    Potts_weight_set parameters, std::string path_to_output,
                    std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    std::cout << "---Running mean-shift and adding super pixel term" <<std::endl;
    crf.addSuperPixel(img,4,2,2000);
    crf.addSuperPixel(img,4,2,200);
    crf.addSuperPixel(img,4,2,20);
    MatrixXf init = crf.unary_init();
    std::vector<perf_measure> traced_perfs_qp;
    std::vector<perf_measure> traced_perfs_qp_nc;
    std::vector<perf_measure> traced_perfs_qp_sp;
    
    //run the inference with the convex problem
    std::cout << "---Finding global optimum, of convex energy function" <<std::endl;
    clock_t start, end;
    start = clock();
    traced_perfs_qp = crf.tracing_qp_inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(init);
    double discretized_energy = crf.assignment_energy(crf.currentMap(init));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(init, size, path_to_output, dataset_name);
    double prev = 0;
    for (auto & it : traced_perfs_qp) {
        prev += it.first;
        std::cout << "(" << prev << "," << it.second << ")";
    }

    //run with non_convex function
    std::cout << "---Finding local optimum, of non-convex energy function" <<std::endl;
    path_to_output.replace(path_to_output.end()-4, path_to_output.end(),"_nc.bmp");
    clock_t start_nc, end_nc;
    start_nc = clock();
    //MatrixXf Q_non_convex = Q;
    //(void) crf.tracing_qp_inference_super_pixels_non_convex(Q_non_convex);
    traced_perfs_qp_nc = crf.tracing_qp_inference_non_convex(init);
    end_nc = clock();
    double timing_non_convex = (double(end_nc-start_nc)/CLOCKS_PER_SEC);
    double final_energy_non_convex = crf.compute_energy(init);
    double discretized_energy_non_convex = crf.assignment_energy(crf.currentMap(init));
    write_down_perf(timing_non_convex, final_energy_non_convex, discretized_energy_non_convex, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(init, size, path_to_output, dataset_name);
    prev = 0;
    for (auto & it : traced_perfs_qp_nc)  {
        prev += it.first;
        std::cout << "(" << prev << "," << it.second << ")";
    }

 /*  
    //we now need to run the code with a non_convex energy function including the super pixels starting at the initial value
    std::cout << "---Finding local optimum, of non-convex energy function with super pixel from initial values" <<std::endl;
    path_to_output.replace(path_to_output.end()-7, path_to_output.end(),"_g_nc_sp.png");
    clock_t start_g_nc_sp, end_g_nc_sp;
    start_g_nc_sp = clock();
    //MatrixXf Q_non_convex_sp = Q;
    //(void) crf.tracing_qp_inference_super_pixels_non_convex(Q_non_convex_sp);
    MatrixXf Q_g_non_convex_sp = crf.unary_init();
    traced_perfs_qp_sp = crf.tracing_qp_inference_super_pixels_non_convex(Q_g_non_convex_sp);
    end_g_nc_sp = clock();
    double timing_g_non_convex_sp = (double(end_g_nc_sp-start_g_nc_sp)/CLOCKS_PER_SEC);
    double final_energy_g_non_convex_sp = crf.compute_energy(Q_g_non_convex_sp);
    double discretized_energy_g_non_convex_sp = crf.assignment_energy(crf.currentMap(Q_g_non_convex_sp));
    write_down_perf(timing_g_non_convex_sp, final_energy_g_non_convex_sp, discretized_energy_g_non_convex_sp, path_to_output);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q_g_non_convex_sp, size, path_to_output, dataset_name);
*/
    delete[] img;
}



void minimize_QP_cccp(std::string path_to_image, std::string path_to_unaries,
                      Potts_weight_set parameters, std::string path_to_output,
                      std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    //crf.compute_kl_divergence();
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    start = clock();
    //Q = crf.qp_inference(init);
    MatrixXf Q = crf.qp_cccp_inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);

    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;
}

void minimize_concave_QP_cccp(std::string path_to_image, std::string path_to_unaries,
                              Potts_weight_set parameters, std::string path_to_output,
                              std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    //crf.compute_kl_divergence();
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.concave_qp_cccp_inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);

    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
}




void minimize_cccp_mean_field(std::string path_to_image, std::string path_to_unaries,
                              Potts_weight_set parameters, std::string path_to_output,
                              std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));

    crf.compute_kl_divergence();
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.cccp_inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);


    // std::cout << "Time taken: " << timing << '\n';
    // std::cout << "KL divergence: " << crf.klDivergence(Q) << '\n';
    // std::cout << "Done with inference"<< '\n';
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;
}

void minimize_LP(std::string path_to_image, std::string path_to_unaries,
                      Potts_weight_set parameters, std::string path_to_output,
                      std::string dataset_name, bool use_cond_grad) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    //crf.compute_kl_divergence();
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    MatrixXf Q = crf.qp_inference(init);
    Q = crf.concave_qp_cccp_inference(Q);

    double timing = -1;
    /*for(int it=0; it<20; it++) {
      std::string partial_out = path_to_output + "-" + std::to_string(it)+ ".bmp";
      Q = crf.lp_inference(Q);
      double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
      double final_energy = crf.compute_energy(Q);
      write_down_perf(timing, final_energy, discretized_energy, partial_out);
      save_map(Q, size, partial_out, dataset_name);
      }/**/
    start = clock();
    srand(start);
    Q = crf.lp_inference(Q, use_cond_grad);
    end = clock();
    timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);
// std::cout << "Time taken: " << timing << '\n';
// std::cout << "Done with inference"<< '\n';
// Perform the MAP estimation on the fully factorized distribution
// and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;
}

void gradually_minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                   Potts_weight_set parameters, std::string path_to_output,
                                   std::string dataset_name) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    MatrixXf init = crf.unary_init();
    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.grad_inference(init);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);

    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    save_map(Q, size, path_to_output, dataset_name);
    delete[] img;
}

void unaries_baseline(std::string path_to_unaries, std::string path_to_output, std::string dataset_name){
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    MatrixXf unaries = load_unary(path_to_unaries, size);
    MatrixXf Q(unaries.rows(), unaries.cols());
    expAndNormalize(Q, -unaries);
    save_map(Q, size, path_to_output, dataset_name);
}


label_matrix minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                 Potts_weight_set parameters) {
    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());

    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));
    MatrixXf init = crf.unary_init();
    MatrixXf Q = crf.inference(init);
    // Perform the MAP estimation on the fully factorized distribution
    // and write the results to an image file with a dumb color code
    delete[] img;
    return get_label_matrix(Q, size);
}
