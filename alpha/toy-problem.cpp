#include "toy-problem.hpp"
#include "brute_force.hpp"
#include "alpha_crf.hpp"

#include <fstream>
#include <iostream>

int randint(int max){
    return std::rand() % max;
}

std::vector<MatrixXf*> get_pairwise(int nb_variables, int nb_features){
    std::vector<MatrixXf*> all_pairwise;
    MatrixXf * pairwise_filter_feat = new MatrixXf(nb_features, nb_variables);
    all_pairwise.push_back(pairwise_filter_feat);

    // This are the features, on which the gaussian distance is going to be defined.
    MatrixXf pairwise_matrix = *pairwise_filter_feat;
    for (int ft=0; ft<nb_features; ft++) {
        for (int var=0; var< nb_variables; var++) {
            pairwise_matrix(ft, var) = randint(10);
        }
    }
    return all_pairwise;
}

MatrixXf get_unaries(int nb_variables, int nb_labels){
    MatrixXf unaries(nb_labels, nb_variables);
    for (int j=0; j<nb_labels; j++) {
        for (int i = 0; i < nb_variables; i++) {
            unaries(j,i) = randint(10);
        }
    }
    return unaries;
}

void original_toy_problem(int argc, char *argv[]) {
    std::srand(1337); // Set the seed
    if (argc < 4) {
		printf("usage: %s w h sigma", argv[0]);
        exit(1);
	}
	int w = atoi(argv[1]);
	int h = atoi(argv[2]);
	int sigma = atoi(argv[3]);
    // lp inference params
	LP_inf_params lp_params;
	if(argc > 4) lp_params.prox_max_iter = atoi(argv[4]);
	if(argc > 5) lp_params.fw_max_iter = atoi(argv[5]);
	if(argc > 6) lp_params.qp_max_iter = atoi(argv[6]);
	if(argc > 7) lp_params.prox_reg_const = atof(argv[7]);
	if(argc > 8) lp_params.dual_gap_tol = atof(argv[8]);
	if(argc > 9) lp_params.qp_tol = atof(argv[9]);
	if(argc > 10) lp_params.best_int = atoi(argv[10]);

    std::cout << "## COMMAND: " << argv[0] << " " << w << " " << h << " " << sigma << " "
        << lp_params.prox_max_iter << " " << lp_params.fw_max_iter << " " << lp_params.qp_max_iter << " "
        << lp_params.prox_reg_const << " " << lp_params.dual_gap_tol << " " << lp_params.qp_tol << " " 
        << lp_params.best_int << std::endl;

    int nb_variables = w*h;
    int nb_labels = 2;    
    float alpha = 10;

    MatrixXf unaries = get_unaries(nb_variables, nb_labels);
    DenseCRF2D crf(w, h, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(sigma, sigma, new PottsCompatibility(10));

    //crf.damp_updates(0.5);
    MatrixXf Q = crf.unary_init();
    double final_energy = crf.compute_energy_true(Q);
    double discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    std::cout << "Before QP: " << final_energy << ", " << discretized_energy << std::endl;
    Q = crf.qp_inference(unaries);
    final_energy = crf.compute_energy_true(Q);
    discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    std::cout << "After QP: " << final_energy << ", " << discretized_energy << std::endl;
//    Q = crf.concave_qp_cccp_inference(Q);
//    final_energy = crf.compute_energy_true(Q);
//    discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
//    std::cout << "After QP-CCCP: " << final_energy << ", " << discretized_energy << std::endl;
    //Q = crf.lp_inference(Q, false);
    Q = crf.lp_inference_prox(Q, lp_params);
    final_energy = crf.compute_energy_true(Q);
    discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    std::cout << "After LP: " << final_energy << ", " << discretized_energy << std::endl;
    //crf.sequential_inference();
    final_energy = crf.compute_energy_true(Q);
    discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    MatrixXf kt_Q = crf.interval_rounding(Q, 100);	// best of 100 KT rounding
	double kt_energy = crf.assignment_energy_true(crf.currentMap(kt_Q));
    printf("Final energy: %lf, best_Q: %lf, kt: %lf\n", final_energy, discretized_energy, kt_energy);

    double ph_energy = 0, bf_energy = 0;
    crf.compare_energies(Q, ph_energy, bf_energy, true, true);
    std::cout << "# lp-pairwise: " << ph_energy << "," << bf_energy << std::endl;
    MatrixXf int_Q = crf.max_rounding(Q);
    crf.compare_energies(int_Q, ph_energy, bf_energy, true, true);
    std::cout << "# int-pairwise: " << ph_energy << "," << bf_energy << std::endl;
}

void compare_bf_ph_energies(int argc, char *argv[]) {
    std::srand(1337); // Set the seed

	if (argc < 6) {
		printf("usage: %s w h sigma qp ph-old subgrad", argv[0]);
        exit(1);
	}
	int w = atoi(argv[1]);
	int h = atoi(argv[2]);
	int sigma = atoi(argv[3]);
	bool qp = atoi(argv[4]);
	bool ph_old = atoi(argv[5]);	
    bool subgrad = false;
    if (argc > 6) subgrad = atoi(argv[6]);

	printf("command: %s %d %d %d %d %d %d\n", argv[0], w, h, sigma, qp, ph_old, subgrad);

    int nb_variables = w*h;
    int nb_labels = 2;    
    float alpha = 10;

    MatrixXf unaries = get_unaries(nb_variables, nb_labels);
    DenseCRF2D crf(w, h, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(sigma, sigma, new PottsCompatibility(1));
    
    // random 
    std::ofstream fout("toy_bf_ph_energy_ran.out");
    MatrixXf Q = unaries;
    Q.fill(0);
    Q.row(0).fill(1);    // all zeros
    int levels = 50;
    int skip = nb_variables/levels;
    int nb_ones = 0;
    double ph_energy = 0, bf_energy = 0;
    int level2 = levels/2;
    // all zeros
    crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old, subgrad);
    fout << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
    std::cout << "#" << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
    int loopmax = 1e6;
    // first half -- choose random ones
    for (int i = 1; i <= level2; ++i) {
        nb_ones = i * skip;
        for (int j = 0; j < nb_ones; ++j) {    // labelling randomly chosen nb_ones elements 
            int p = randint(nb_variables);
            int count = 0;
            while(Q(1, p) == 1 && count <= loopmax) {    // if already labelled choose another
                p = randint(nb_variables);
                ++count;
            }
            Q(0, p) = 0;
            Q(1, p) = 1;
        }
        crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old, subgrad);
        fout << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
        std::cout << "#" << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
        Q.row(0).fill(1);    
        Q.row(1).fill(0);    // all zeros
    }
    
    // second half - choose random zeros
    Q.row(0).fill(0);
    Q.row(1).fill(1);    // all ones
    level2 = levels - level2;
    for (int i = level2-1; i >= 1; --i) {
        int nb_zeros = i * skip;
        for (int j = 0; j < nb_zeros; ++j) {    // labelling randomly chosen nb_zeros elements 
            int p = randint(nb_variables);
            int count = 0;
            while(Q(1, p) == 0 && count <= loopmax) {    // if already labelled choose another
                p = randint(nb_variables);
                ++count;
            }
            Q(0, p) = 1;
            Q(1, p) = 0;
        }
        nb_ones = nb_variables - nb_zeros;
        crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old, subgrad);
        fout << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
        std::cout << "#" << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
        Q.row(0).fill(0);
        Q.row(1).fill(1);    // all ones
    }
    // all ones    
    nb_ones = nb_variables;
    crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old, subgrad);
    fout << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
    std::cout << "#" << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
    fout.close();

    for (int i = 0; i < Q.cols(); ++i) {
        for (int j = 0; j < Q.rows(); ++j) {
            int r = randint(10);
            Q(j, i) = float(r)/10.0;
        }
    }
    crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old, subgrad);
    std::cout << "#random-frac: " << ph_energy << "," << bf_energy << std::endl;
}

int main(int argc, char *argv[])
{
    //original_toy_problem(argc, argv);

    compare_bf_ph_energies(argc, argv);

    return 0;
}
