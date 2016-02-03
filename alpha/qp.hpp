#pragma once
#include "eigen_utils.hpp"

void descent_direction(Eigen::MatrixXf & out, const Eigen::MatrixXf & grad );

float pick_lambda_eig_to_convex(const MatrixXf & lbl_compatibility);
