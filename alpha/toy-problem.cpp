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

void original_toy_problem() {
    std::srand(1337); // Set the seed
    int nb_variables = 100;
    int nb_labels = 5;
    int nb_features = 1;
    float alpha = 10;

    MatrixXf unaries = get_unaries(nb_variables, nb_labels);
    //std::vector<MatrixXf*> all_pairwise = get_pairwise(nb_variables, nb_features);

    AlphaCRF crf(1, nb_variables, nb_labels, alpha); // width, height, nb_labels, alpha
    crf.keep_track_of_steps();
    //crf.compute_exact_marginals();

    crf.setUnaryEnergy(unaries);
    for (int filter = 0; filter < nb_features; filter++) {
        // crf.addPairwiseEnergy(*(all_pairwise[filter]), new PottsCompatibility(1));
        crf.addPairwiseGaussian(5, 5, new PottsCompatibility(1));
    }

    //crf.damp_updates(0.5);
    MatrixXf Q = unaries;
    Q = crf.qp_inference(unaries);
    Q = crf.concave_qp_cccp_inference(Q);
    //Q = crf.lp_inference(Q, false);
    //crf.sequential_inference();
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    printf("Final: %lf, Proj: %lf\n", final_energy, discretized_energy);
}

void compare_bf_ph_energies(int argc, char *argv[]) {
    std::srand(1337); // Set the seed

	if (argc < 6) {
		printf("usage: %s w h sigma qp ph-old", argv[0]);
        exit(1);
	}
	int w = atoi(argv[1]);
	int h = atoi(argv[2]);
	int sigma = atoi(argv[3]);
	bool qp = atoi(argv[4]);
	bool ph_old = atoi(argv[5]);	

	printf("command: %s %d %d %d %d %d\n", argv[0], w, h, sigma, qp, ph_old);

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
    crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old);
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
        crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old);
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
        crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old);
        fout << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
        std::cout << "#" << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
        Q.row(0).fill(0);
        Q.row(1).fill(1);    // all ones
    }
    // all ones    
    nb_ones = nb_variables;
    crf.compare_energies(Q, ph_energy, bf_energy, qp, ph_old);
    fout << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
    std::cout << "#" << nb_ones << "," << ph_energy << "," << bf_energy << std::endl;
    fout.close();
}


int main(int argc, char *argv[])
{
    //original_toy_problem();

    compare_bf_ph_energies(argc, argv);

    return 0;
}
