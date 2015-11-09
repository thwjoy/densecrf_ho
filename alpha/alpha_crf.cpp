#include "alpha_crf.hpp"
//#include "densecrf.h"
#include <iostream>


//////////////////////////////////
// Constructors and Destructors //
//////////////////////////////////

AlphaCRF::AlphaCRF(int W, int H, int M, float alpha) : DenseCRF2D(W, H, M), alpha(alpha){
}
AlphaCRF::~AlphaCRF(){}


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
        unary = getUnaryEnergy()->get();
        std::cout << unary.rows() <<std::endl;
        std::cout << "good unaries obtained" << '\n';
    }


    std::cout << "Initializing the approximating distribution with the unaries." << '\n';
    expAndNormalize( Q, -unary);
    std::cout << "Got initial estimates of the distribution" << '\n';


    return Q;
}
