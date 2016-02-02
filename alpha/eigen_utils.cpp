#include "eigen_utils.hpp"

bool all_close_to_zero(const VectorXf & vec, float ref){
    for (int i = 0; i<vec.size() ; i++) {
        if(vec(i)> ref or vec(i)<-ref){
            return false;
        }
    }
    return true;
}

bool all_positive(const VectorXf & vec){
    for (int i=0; i < vec.size(); i++) {
        if(vec(i)< 0){
            return false;
        }
    }
    return true;
}

bool all_strict_positive(const VectorXf & vec){
    for (int i=0; i < vec.size(); i++) {
        if(vec(i)<= 0){
            return false;
        }
    }
    return true;
}



bool valid_probability(const MatrixXf & proba){
    for (int i=0; i<proba.cols(); i++) {
        if (not all_positive(proba.col(i))) {
            return false;
        }
        if (fabs(proba.col(i).sum()-1)>1e-5) {
            return false;
        }
    }
    return true;
}

typeP dotProduct(const MatrixXf & M1, const MatrixXf & M2, MatrixP & temp){
    // tmp is an already allocated and well dimensioned temporary
    // matrix so that we don't need to allocate a new one. This may
    // very well be premature optimisation.
    temp = (M1.cast<typeP>()).cwiseProduct(M2.cast<typeP>());
    return temp.sum();
}
