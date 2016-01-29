#include <Eigen/Core>

using namespace Eigen;


bool all_close_to_zero(const VectorXf & vect, float ref);
bool all_positive(const VectorXf & vect);
bool all_strict_positive(const VectorXf & vect);
bool valid_probability(const MatrixXf & proba);
