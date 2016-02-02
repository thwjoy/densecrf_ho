#include <Eigen/Core>

using namespace Eigen;


typedef MatrixXd MatrixP;
typedef VectorXd VectorP;
typedef double typeP;

bool all_close_to_zero(const VectorXf & vect, float ref);
bool all_positive(const VectorXf & vect);
bool all_strict_positive(const VectorXf & vect);
bool valid_probability(const MatrixXf & proba);

typeP dotProduct(const MatrixXf & M1, const MatrixXf & M2, MatrixP & temp);
