#include "brute_force.hpp"
#include <algorithm>
#include <limits>
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

double compute_partition_function(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise_feats,
                                  const std::vector<MatrixXf> & label_compatibility) {
    int M_ = unaries.rows();
    int N_ = unaries.cols();
    double Z = 0;
    double conf_proba;
    VectorXi current_labeling(N_);
    current_labeling.fill(0);
    bool all_conf_done=false;

    while(not all_conf_done){
        conf_proba = compute_probability(current_labeling, unaries, pairwise_feats, label_compatibility, 1);
        Z += conf_proba;
        all_conf_done = get_next_labeling(current_labeling, M_);
    }

    return Z;
}

double compute_alpha_divergence(const MatrixXf & unaries,const std::vector<MatrixXf> & pairwise_feats,
                                const std::vector<MatrixXf> & label_compatibility,
                                const MatrixXf & approximation, float alpha){

    int M_ = unaries.rows();
    int N_ = unaries.cols();

    // Compute the partition function of P
    double Z = compute_partition_function(unaries, pairwise_feats, label_compatibility);

    // We have the proper value of Z
    VectorXi current_labeling(N_);
    current_labeling.fill(0);
    bool all_conf_done=false;
    double approx_proba;
    double conf_proba;
    double D_alpha = 0;

    while (not all_conf_done) {
        conf_proba = compute_probability(current_labeling, unaries, pairwise_feats, label_compatibility, Z);
        approx_proba = compute_approx_proba(current_labeling, approximation);

        D_alpha += -pow(conf_proba, alpha) * pow(approx_proba, 1-alpha) +
            alpha* conf_proba +
            (1-alpha) * approx_proba;
        //D_alpha += -pow(conf_proba, alpha) * pow(approx_proba, 1-alpha); // simpler version
        all_conf_done = get_next_labeling(current_labeling, M_);
    }
    D_alpha = (1/(alpha * (1-alpha))) * D_alpha;

    std::cout <<"Alpha-divergence:\t" <<D_alpha << '\n';
    return D_alpha;
}

double compute_kl_div(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise_feats,
                      const std::vector<MatrixXf> & label_compatibility, const MatrixXf & approximation){
    int M_ = unaries.rows();
    int N_ = unaries.cols();


    double Z = compute_partition_function(unaries, pairwise_feats, label_compatibility);

    VectorXi current_labeling(N_);
    current_labeling.fill(0);
    bool all_conf_done = false;
    double approx_proba;
    double conf_proba;
    double kl = 0;
    double eps = std::numeric_limits<double>::epsilon();

    while(not all_conf_done) {
        conf_proba = compute_probability(current_labeling, unaries, pairwise_feats, label_compatibility, Z);
        approx_proba = compute_approx_proba(current_labeling, approximation);
        double term = approx_proba * log(std::max(approx_proba/std::max(conf_proba, eps), eps));
        kl = kl + term;
        all_conf_done = get_next_labeling(current_labeling, M_);
    }
    std::cout << "KL-divergence:\t " << kl << '\n';
    return kl;
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


double compute_probability(const VectorXi& labeling, const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise_feats, const std::vector<MatrixXf> & label_compatibility, double Z){
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
        for(int filter =0; filter<pairwise_feats.size(); ++filter){
            VectorXf first_feat = pairwise_feats[filter].col(i);
            for (int j=i; j< labeling.size(); j++) {
                float weight = label_compatibility[filter](i,j);
                VectorXf second_feat = pairwise_feats[filter].col(j);
                energy += weight * exp(-(first_feat-second_feat).squaredNorm());
            }
        }
    }
    return exp(-energy)/Z;
}


MatrixXf brute_force_marginals(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise_feats, const std::vector<MatrixXf> & label_compatibility) {
    // Compute the marginals of the distribution that has:
    // - Unaries as unary terms
    // - Pairwise terms defined by pairwise_feats (as points, already normalized by their stdev)
    //   and label compatibility as weights

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
        conf_un_proba = compute_probability(current_labeling, unaries, pairwise_feats, label_compatibility, 1);
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
