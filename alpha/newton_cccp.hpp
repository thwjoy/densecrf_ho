#include "eigen_utils.hpp"

void newton_cccp(VectorXf & state, const VectorXf & cste, float lamda_eig);
float pick_lambda_eig_to_concave(const MatrixXf & lbl_compatibility);
