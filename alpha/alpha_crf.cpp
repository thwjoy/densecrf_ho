#include "alpha_crf.hpp"
//#include "densecrf.h"
#include <iostream>


//////////////////////////////////
// Constructors and Destructors //
//////////////////////////////////

AlphaCRF::AlphaCRF(int W, int H, int M, float alpha) : DenseCRF2D(W, H, M), alpha(alpha){
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
    std::cout << "Starting inference to minimize alpha-divergence." << '\n';

    MatrixXf Q(M_, N_), surr_unaries(M_, N_), unary(M_, N_);
    std::vector<PairwisePotential*> surr_pairwise;



    if(!unary_){
        unary.fill(0);
    } else {
        std::cout << unary_ << '\n';
        std::cout << getUnaryEnergy() << '\n';

        std::cout << "Getting proper unaries"  << '\n';
        unary = unary_->get();
        std::cout << unary.rows() <<std::endl;
        std::cout << "good unaries obtained" << '\n';
    }


    std::cout << "Initializing the approximating distribution with the unaries." << '\n';
    expAndNormalize( Q, -unary);
    std::cout << "Got initial estimates of the distribution" << '\n';


    // Compute the factors for the approximate distribution
    //// Unaries
    MatrixXf true_unary_part = alpha* unary;
    MatrixXf approx_part = (1-alpha) * Q.array().log() * -1;
    proxy_unary = true_unary_part + approx_part;
    //// Pairwise term are created when we set up the CRF because they
    //// are going to remain the same




    return Q;
}
