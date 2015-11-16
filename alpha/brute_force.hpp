#include <Eigen/Core>

using namespace Eigen;

double compute_alpha_divergence(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise, const std::vector<float> & pairwise_weight, const MatrixXf & approximation, float alpha);

double compute_probability(const VectorXi& labeling,const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise, const std::vector<float> & pairwise_weights, double Z);

double compute_approx_proba(const VectorXi& labeling, const MatrixXf & approximation);

