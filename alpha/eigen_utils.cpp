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

void sortRows(const MatrixXf & M, MatrixXi & ind) {
    int nbrRows = M.rows();
    int nbrCols = M.cols();
    // Initialize the indices matrix
    for(int i=0; i<ind.cols(); ++i) {
        ind.col(i).fill(i);
    }

    // we need indices for the row to be contiguous in memory
    ind.transposeInPlace();

    for(int i=0; i<nbrRows; ++i) {
        std::sort(ind.data()+i*nbrCols,
                  ind.data()+(i+1)*nbrCols,
                  [&M, i](int a, int b){return M(i, a)>M(i, b);});
    }

    ind.transposeInPlace();
}

void sortCols(const MatrixXf & M, MatrixXi & ind) {
    // Sort each row of M independantly and store indices in ind
    int nbrRows = M.rows();
    int nbrCols = M.cols();
    for(int i=0; i<ind.rows(); ++i) {
        ind.row(i).fill(i);
    }

    // indices for the cols are already contiguous in memory
    for(int i=0; i<nbrCols; ++i) {
        std::sort(ind.data()+i*nbrRows,
                  ind.data()+(i+1)*nbrRows,
                  [&M, i](int a, int b){
                    return M(a, i)>M(b, i);
                  });
    }

}