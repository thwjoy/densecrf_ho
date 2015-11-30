#include "brute_force.hpp"
#include <iostream>

bool get_next_labeling(VectorXi & labeling, int nb_labels){
    /**
     * Increment "lexically" the vector of labeling and return a
     * boolean, indicating whether or not this was the last labeling
     * to look at.
     */
    ++labeling(0);
    for (int i=0; i < labeling.size() -1; i++) {
        if (labeling(i)==nb_labels) {
            labeling(i)=0;
            ++labeling(i+1);
        }
    }
    return labeling(labeling.size()-1) == nb_labels;
}


double compute_alpha_divergence(const MatrixXf & unaries,const std::vector<MatrixXf> & pairwise, const std::vector<float> & pairwise_weight,const MatrixXf & approximation, float alpha){

    int M_ = unaries.rows();
    int N_ = unaries.cols();


    // Compute the partition function of P
    double Z = 0;
    double conf_proba;
    VectorXi current_labeling(N_);
    current_labeling.fill(0);
    bool all_conf_done=false;

    while(not all_conf_done){
        conf_proba = compute_probability(current_labeling, unaries, pairwise, pairwise_weight, 1);
        Z += conf_proba;
        all_conf_done = get_next_labeling(current_labeling, M_);
    }
    // We have the proper value of Z
    current_labeling.fill(0);
    all_conf_done=false;
    double approx_proba;
    double D_alpha = 0;

    while (not all_conf_done) {
        conf_proba = compute_probability(current_labeling, unaries, pairwise, pairwise_weight, Z);
        approx_proba = compute_approx_proba(current_labeling, approximation);

        D_alpha += -pow(conf_proba, alpha) * pow(approx_proba, 1-alpha) +
            alpha* conf_proba +
            (1-alpha) * approx_proba;
        //D_alpha += -pow(conf_proba, alpha) * pow(approx_proba, 1-alpha);
        all_conf_done = get_next_labeling(current_labeling, M_);
    }
    D_alpha = (1/(alpha * (1-alpha))) * D_alpha;

    std::cout << D_alpha << '\n';
    return D_alpha;
}

double compute_approx_proba(const VectorXi & labeling, const MatrixXf & approximation) {
    /**
     * Computing the approximate probability is just a matter of
     * multiplying the marginals
     */
    double approx_proba = 1;
    for (int i=0; i<labeling.size(); i++) {
        approx_proba *= approximation(labeling(i), i);
    }
    return approx_proba;
}


double compute_probability(const VectorXi& labeling, const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise, const std::vector<float> & pairwise_weights, double Z){
    /**
     * labeling is a vector indicating which label each variable takes,
     * unaries is a M_ by N_ matrix containing the unaries value
     * pairwise is a list of matrix nb_feat by N_ containing features of pairwise filters
     * pairwise weights contains the weight associated with these filters
     * Z is the partition function
     */
    double energy = 0;
    for(int i=0; i<labeling.size(); i++){
        energy = energy + unaries(labeling(i), i);
        for(int filter =0; filter<pairwise.size(); ++filter){
            float weight = pairwise_weights[filter];
            VectorXf first_feat = pairwise[filter].col(i);
            for (int j=i; j< labeling.size(); j++) {
                if (labeling(i)!=labeling(j)) {
                    VectorXf second_feat = pairwise[filter].col(j);
                    energy += weight * exp(-(first_feat-second_feat).squaredNorm());
                }

            }

        }
    }
    return exp(-energy)/Z;
}


MatrixXf brute_force_marginals(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise, const std::vector<float> & pairwise_weight) {
    int M_ = unaries.rows();
    int N_ = unaries.cols();

    MatrixXf marginals(M_, N_);
    marginals.fill(0);

    double conf_un_proba;
    VectorXi current_labeling(N_);
    current_labeling.fill(0);
    bool all_conf_done=false;
    double norm = 0;

    // Get the unnormalized marginals.
    while(not all_conf_done){
        conf_un_proba = compute_probability(current_labeling, unaries, pairwise, pairwise_weight, 1);
        for (int i=0; i < N_; i++) {
            marginals(current_labeling(i), i) += conf_un_proba;
        }
        norm += conf_un_proba;
        all_conf_done = get_next_labeling(current_labeling, M_);
    }

    // Normalize the marginals (shouldn't be necessary)
    marginals.array() /= norm;

    return marginals;
}
