#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "alpha_crf.hpp"
#include "color_to_label.hpp"
#include "inference.hpp"

using namespace Eigen;

#define DEFAULT_SIZE -1
#define BINARY false

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

void minimize_prox_LP(std::string path_to_image, std::string path_to_unaries,
                      Potts_weight_set parameters, std::string path_to_output,
                      std::string dataset_name, int argc, char* argv[]) {

    // lp inference params
	LP_inf_params lp_params;
	if(argc > 1) lp_params.prox_max_iter = atoi(argv[1]);
	if(argc > 2) lp_params.fw_max_iter = atoi(argv[2]);
	if(argc > 3) lp_params.qp_max_iter = atoi(argv[3]);
	if(argc > 4) lp_params.prox_reg_const = atof(argv[4]);
	if(argc > 5) lp_params.dual_gap_tol = atof(argv[5]);
	if(argc > 6) lp_params.qp_tol = atof(argv[6]);
	if(argc > 7) lp_params.best_int = atoi(argv[7]);
    lp_params.prox_energy_tol = lp_params.dual_gap_tol;
	if(argc > 8) lp_params.prox_energy_tol = atof(argv[8]);

    std::cout << "## COMMAND: " << argv[0] << " " 
        << lp_params.prox_max_iter << " " << lp_params.fw_max_iter << " " << lp_params.qp_max_iter << " "
        << lp_params.prox_reg_const << " " << lp_params.dual_gap_tol << " " << lp_params.qp_tol << " " 
        << lp_params.best_int << " " << lp_params.prox_energy_tol << std::endl;

    img_size size = {DEFAULT_SIZE, DEFAULT_SIZE};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(path_to_unaries, size);
    unsigned char * img = load_image(path_to_image, size);

#if BINARY
    int bg, fg;
    if (dataset_name == "Pascal2010") {
        bg = 0;
        fg = 1;
    } else if (dataset_name=="MSRC") {
        bg = 1;
        fg = 3;
    }
    MatrixXf tmp(2, unaries.cols());
    tmp.row(0) = unaries.row(bg);
    tmp.row(1) = unaries.row(fg);
    unaries = tmp;
#endif

    // Load a crf
    DenseCRF2D crf(size.width, size.height, unaries.rows());
    std::cout << "CRF: W = " << size.width << ", H = " << size.height << ", L = " << unaries.rows() << std::endl;

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
    start = clock();
    srand(start);
    
    Q = crf.lp_inference_prox(Q, lp_params);
    end = clock();
    timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    write_down_perf(timing, final_energy, discretized_energy, path_to_output);
    double final_energy_true = crf.compute_energy_true(Q);
    double discretized_energy_true = crf.assignment_energy_true(crf.currentMap(Q));
    std::cout << "QP: " << final_energy << ", int: " << discretized_energy << std::endl;
    std::cout << "#TRUE QP: " << final_energy_true << ", LP: " << crf.compute_energy_LP(Q) 
        << ", int: " << discretized_energy_true << std::endl;

    MatrixXf int_Q = crf.max_rounding(Q);

#if BINARY    
    double ph_energy = 0, bf_energy = 0;
    crf.compare_energies(Q, ph_energy, bf_energy, false, false, true);
    std::cout << "# lp-pairwise: " << ph_energy << "," << bf_energy << std::endl;
    crf.compare_energies(int_Q, ph_energy, bf_energy, false, false, true);
    std::cout << "# int-pairwise: " << ph_energy << "," << bf_energy << std::endl;
#endif

    std::cout << "# int-LP-total: " << crf.compute_energy_LP(int_Q) << ", int-QP-total: " 
        << crf.compute_energy_true(int_Q) << std::endl;

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
