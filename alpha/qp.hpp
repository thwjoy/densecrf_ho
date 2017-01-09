#pragma once
#include "eigen_utils.hpp"

void descent_direction(Eigen::MatrixXf & out, const Eigen::MatrixXf & grad );
void super_descent_direction(MatrixXf & out, const MatrixXf & grad);
void compute_hyper_params(const MatrixXf & Q, MatrixXf & params,const MatrixXf & super_pix_classifier);
float pick_lambda_eig_to_convex(const MatrixXf & lbl_compatibility);
