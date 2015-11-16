#include "alpha_crf.hpp"
#include <iostream>




//////////////////////////////////
// Constructors and Destructors //
//////////////////////////////////

AlphaCRF::AlphaCRF(int W, int H, int M, float alpha) : DenseCRF2D(W, H, M), alpha(alpha) {
}
AlphaCRF::~AlphaCRF(){}

// Overload the addition of the pairwise energy so that it adds the
// proxy-term with the proper weight
void AlphaCRF::addPairwiseEnergy(const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type, NormalizationType normalization_type){
    assert(features.cols() == N_);
    function->setParameters( alpha * function->parameters());
    DenseCRF::addPairwiseEnergy( new PairwisePotential( features, function, kernel_type, normalization_type));
    if (monitor_mode) {
        pairwise_weights.push_back(function->parameters());
        pairwise_features.push_back(&features);
    }
}

void AlphaCRF::keep_track_of_steps(){
    monitor_mode = true;
};

////////////////////
// Inference Code //
////////////////////

MatrixXf AlphaCRF::inference(){
    D("Starting inference to minimize alpha-divergence.");
    // Q contains our approximation, unary contains the true
    // distribution unary, approx_Q is the meanfield approximation of
    // the proxy-distribution
    MatrixXf Q(M_, N_), unary(M_, N_), approx_Q(M_, N_), Q_old(M_, N_), approx_Q_old(M_,N_);
    // tmp1 and tmp2 are matrix to gather intermediary computations
    MatrixXf tmp1(M_, N_), tmp2(M_, N_);


    if(!unary_){
        unary.fill(0);
    } else {
        unary = unary_->get();
    }
    D("Initializing the approximating distribution");
    expAndNormalize( Q, -unary); // Initialization by the unaries
    //Q.fill(1/(float)M_); // Initialization to a uniform distribution
    D("Got initial estimates of the distribution");

    Q_old = Q;
    bool continue_minimizing_alpha_div = true;
    float Q_change;
    int nb_approximate_distribution = 0;
    while(continue_minimizing_alpha_div){

        if (monitor_mode) {
            float ad = compute_alpha_divergence(unary, pairwise_features, pairwise_weights, Q, alpha);
            alpha_divergences.push_back(ad);
        }

        D("Constructing proxy distributions");
        // Compute the factors for the approximate distribution
        //Unaries
        MatrixXf true_unary_part = alpha* unary;
        MatrixXf approx_part = -1 * (1-alpha) * Q.array().log();
        proxy_unary = true_unary_part + approx_part;

        //// Pairwise term are created when we set up the CRF because they
        //// are going to remain the same
        D("Done constructing the proxy distribution");;

        D("Starting to estimate the marginals of the distribution");
        // Starting value.
        expAndNormalize(approx_Q, -proxy_unary); // Initialization by the unaries
        //approx_Q.fill(1/(float)M_); // Uniform initialization

        // Setup the checks for convergence.
        bool continue_estimating_marginals = true;
        approx_Q_old = approx_Q;
        float marginal_change;
        int nb_marginal_estimation = 0;
        while(continue_estimating_marginals) {
            // Perform one meanfield iteration to update our approximation
            mf_for_marginals(approx_Q, tmp1, tmp2);
            // Evaluate how much our distribution changed
            marginal_change = (approx_Q - approx_Q_old).squaredNorm();
            // If we stopped changing a lot, stop the loop and
            // consider we have some good marginals
            approx_Q_old = approx_Q;
            continue_estimating_marginals = (marginal_change > 0.001);
            ++ nb_marginal_estimation;
        }
        std::cout << nb_marginal_estimation << '\t';
        D("Finished MF marginals estimation");

        D("Estimate the update rule parameters");
        tmp1 = Q.array().pow(alpha-1);
        tmp2 = tmp1.cwiseProduct(approx_Q);
        tmp2 = tmp2.array().pow(1/alpha);
        expAndNormalize(Q, tmp2);
        D("Updated our approximation");

        Q_change = (Q_old - Q).squaredNorm();
        Q_old = Q;
        continue_minimizing_alpha_div = (Q_change > 0.001);
        ++nb_approximate_distribution;
    }
    std::cout << "\n Nb of approximate distribution constructed:" << nb_approximate_distribution << '\n';
    D("Done with alpha-divergence minimization");
    return Q;
}

// Reuse the same tempvariables at all step.
    void AlphaCRF::mf_for_marginals(MatrixXf & approx_Q, MatrixXf & tmp1, MatrixXf & tmp2) {
        tmp1 = -proxy_unary;

        for (int i=0; i<pairwise_.size(); i++) {
            pairwise_[i]->apply(tmp2, approx_Q);
            tmp1 -= tmp2;
        }

        expAndNormalize(approx_Q, tmp1);
    }
