#include <Eigen/Core>

using namespace Eigen;


typedef MatrixXd MatrixP;
typedef VectorXd VectorP;
typedef double typeP;

bool all_close_to_zero(const VectorXf & vect, float ref);
bool all_positive(const VectorXf & vect);
bool all_strict_positive(const VectorXf & vect);
bool valid_probability(const MatrixXf & proba);
bool valid_probability_debug(const MatrixXf & proba);
void clamp_and_normalize(VectorXf & prob);

typeP dotProduct(const MatrixXf & M1, const MatrixXf & M2, MatrixP & temp);
void sortRows(const MatrixXf & M, MatrixXi & ind);
void sortCols(const MatrixXf & M, MatrixXi & ind);
float infiniteNorm(const MatrixXf & M);
