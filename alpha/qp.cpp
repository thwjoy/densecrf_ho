#include "qp.hpp"
#include <iostream>
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

void descent_direction_z(MatrixXf & out, const MatrixXf & grad){
    out.resize(grad.rows(), grad.cols());
    out.fill(0);
    for (int row = 0; row < grad.rows(); row++) {
        for (int col = 0; col < grad.cols(); col++) {
            out(row,col) = (grad(row,col) < 0) ? 1 : 0;
        }
    }
}


void compute_hyper_params(const MatrixXf & Q, MatrixXf & params,const MatrixXf & super_pix_classifier) {
    int regions = super_pix_classifier.rows();
    int pixles = super_pix_classifier.cols();
    int lables = Q.rows();
    float prod;
    int sum;
    for (int lab = 0; lab < lables; lab++ ) {
        prod = 1;
        for (int i = 0; i < regions; i++) {
            sum = 0;
            for (int pix = 0; pix < pixles; pix++) {           
                sum += ((super_pix_classifier(i, pix)) * (1 - Q(lab,pix))); //we want to do this for all labels
            }
            prod = prod * sum;
        }
        params.row(lab).fill((prod == 0) ? 1 : 0);
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


