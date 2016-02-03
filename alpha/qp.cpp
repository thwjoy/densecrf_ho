#include "qp.hpp"
#include <Eigen/Eigenvalues>

void descent_direction(MatrixXf & out, const MatrixXf & grad){
    out.resize(grad.rows(), grad.cols());
    out.fill(0);
    int N =  grad.cols();

    int m;
    for (int i=0; i < N; i++) {
        grad.col(i).minCoeff(&m);
        out(m,i) = 1;
    }
}


float pick_lambda_eig_to_convex(const MatrixXf & lbl_compatibility){
    // We assume that the label compatibility matrix is symmetric in
    // order to use eigens eigenvalue code.
    VectorXf eigs = lbl_compatibility.selfadjointView<Eigen::Upper>().eigenvalues();
    float lambda_eig = eigs.minCoeff();
    lambda_eig = lambda_eig < 0 ? lambda_eig: 0;
    return lambda_eig;
}
