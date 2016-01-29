#include <Eigen/Core>

bool all_close_to_zero(const Eigen::VectorXf & vect, float ref);
bool all_positive(const Eigen::VectorXf & vect);
bool all_strict_positive(const Eigen::VectorXf & vect);
bool valid_probability(const Eigen::MatrixXf & proba);
