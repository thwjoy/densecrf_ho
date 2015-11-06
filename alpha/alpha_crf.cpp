#include "alpha_crf.hpp"
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
    return MatrixXf(1,1);
}
