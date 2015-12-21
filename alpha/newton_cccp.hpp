#include <Eigen/Core>

using namespace Eigen;

void newton_cccp(VectorXf & state, const VectorXf & cste, float lamda_eig);
float pick_lambda_eig(MatrixXf const & lbl_compatibility);
