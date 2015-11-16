#include "toy-problem.hpp"
#include "brute_force.hpp"
#include "alpha_crf.hpp"


int randint(int max){
    return std::rand() % max;
}

std::vector<MatrixXf*> get_pairwise(int nb_variables, int nb_features){
    std::vector<MatrixXf*> all_pairwise;
    MatrixXf * pairwise_filter_feat = new MatrixXf(nb_features, nb_variables);
    all_pairwise.push_back(pairwise_filter_feat);


    MatrixXf pairwise_matrix = *pairwise_filter_feat;
    for (int ft=0; ft<nb_features; ft++) {
        for (int var=0; var< nb_variables; var++) {
            pairwise_matrix(ft, var) = randint(5);
        }
    }
    return all_pairwise;
}

MatrixXf get_unaries(int nb_variables, int nb_labels){
    MatrixXf unaries(nb_labels, nb_variables);
    for (int j=0; j<nb_labels; j++) {
        for (int i = 0; i < nb_variables; i++) {
            unaries(j,i) = randint(5);
        }
    }
    return unaries;
}

int main(int argc, char *argv[])
{
    std::srand(time(NULL)); // Set the seed
    int nb_variables = 5;
    int nb_labels = 5;
    int nb_features = 5;
    float alpha = 10;

    MatrixXf unaries = get_unaries(nb_variables, nb_features);
    std::vector<MatrixXf*> all_pairwise = get_pairwise(nb_variables, nb_labels);

    AlphaCRF crf(1, nb_variables, nb_labels, alpha);
    crf.keep_track_of_steps();

    crf.setUnaryEnergy(unaries);
    for (int filter = 0; filter < all_pairwise.size() ; filter++) {
        crf.addPairwiseEnergy(*(all_pairwise[filter]), new PottsCompatibility(10));
    }
    crf.damp_updates(0.5);

    crf.inference();

    return 0;
}
