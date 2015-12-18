#include <Eigen/Core>

using namespace Eigen;

double compute_alpha_divergence(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise,
                                const std::vector<MatrixXf> & label_compatibility, const MatrixXf & approximation,
                                float alpha);
double compute_kl_div(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise,
                                const std::vector<MatrixXf> & label_compatibility,
                                const MatrixXf & approximation);
double compute_probability(const VectorXi& labeling,const MatrixXf & unaries,
                           const std::vector<MatrixXf> & pairwise,
                           const std::vector<MatrixXf> & label_compatibility, double Z);
double compute_approx_proba(const VectorXi& labeling, const MatrixXf & approximation);
MatrixXf brute_force_marginals(const MatrixXf & unaries, const std::vector<MatrixXf> & pairwise,
                               const std::vector<MatrixXf> & pairwise_weight);
