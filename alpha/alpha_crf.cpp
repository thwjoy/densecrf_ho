#include "alpha_crf.hpp"
//#include "densecrf.h"
#include <iostream>


//////////////////////////////////
// Constructors and Destructors //
//////////////////////////////////

AlphaCRF::AlphaCRF(int W, int H, int M, float alpha, int nb_mf_marg) : DenseCRF2D(W, H, M), alpha(alpha){
}
AlphaCRF::~AlphaCRF(){}

// Overload the addition of the pairwise energy so that it adds the
// proxy-term with the proper weight
void AlphaCRF::addPairwiseEnergy(const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type, NormalizationType normalization_type){
    assert(features.cols() == N_);
    function->setParameters( alpha * function->parameters());
    DenseCRF::addPairwiseEnergy( new PairwisePotential( features, function, kernel_type, normalization_type));
}

////////////////////
// Inference Code //
////////////////////

MatrixXf AlphaCRF::inference(int nb_iterations){
    std::cout << "Starting inference to minimize alpha-divergence." << "\n\n";

    // Q contains our approximation, unary contains the true
    // distribution unary, approx_Q is the meanfield approximation of
    // the proxy-distribution
    MatrixXf Q(M_, N_), unary(M_, N_), approx_Q(M_, N_);
    // tmp1 and tmp2 are matrix to gather intermediary computations
    MatrixXf tmp1(M_, N_), tmp2(M_, N_);


    if(!unary_){
        unary.fill(0);
    } else {
        unary = unary_->get();
    }

    std::cout << "Initializing the approximating distribution with the unaries." << '\n';
    expAndNormalize( Q, -unary);
    std::cout << "Got initial estimates of the distribution" << "\n\n";

    for (int nb_iter=0; nb_iter < nb_iterations; nb_iter++) {

        std::cout << "Constructing proxy distributions" << '\n';
        // Compute the factors for the approximate distribution
        //Unaries
        MatrixXf true_unary_part = alpha* unary;
        MatrixXf approx_part = (1-alpha) * Q.array().log() * -1;
        proxy_unary = true_unary_part + approx_part;
        //// Pairwise term are created when we set up the CRF because they
        //// are going to remain the same // TODO: Verify that they are created properly.
        std::cout << "Done constructing the proxy distribution" << "\n\n";

        std::cout << "Starting to estimate the marginals of the distribution" << '\n';
        approx_Q = Q;
        for (int marg_est_cnt=0; marg_est_cnt<nb_mf_marg; marg_est_cnt++) {
            mf_for_marginals(approx_Q, tmp1, tmp2);
        }
        std::cout << "Finished MF marginals estimation" << "\n\n";

        std::cout << "Estimate the update rule parameters" << '\n';
        tmp1 = Q.array().pow(alpha-1);
        tmp2 = tmp1.cwiseProduct(approx_Q);
        tmp2 = tmp2.array().pow(1/alpha);
        expAndNormalize(Q, tmp2);
        std::cout << "Updated our approximation" << "\n\n";


    }



    return Q;
}

// Reuse the same tempvariables at all step.
void AlphaCRF::mf_for_marginals(MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2) {
    tmp1 = -proxy_unary; // TODO: check that this is not cloning-linking it.

    for (int i=0; i<pairwise_.size(); i++) {
        pairwise_[i]->apply(tmp2, Q);
        tmp1 -= tmp2;
    }

    expAndNormalize(Q, tmp1);
}
