#include "densecrf.h"
#include "util.hpp"

class AlphaCRF: public DenseCRF2D {
protected:

    //factors of the proxy-distribution that we are minimizing for.
    MatrixXf proxy_unary;

    // Alpha for which we minimize the alpha-divergence.
    const float alpha;

    // Use damping or not, when updating the approximation
    bool use_damping = false;
    float damping_factor = 1e-3;

    // Brute-force the marginals or update them with Alpha-divergences
    bool exact_marginals_mode = false;
    // History of all the alpha divergences that we went through
    bool monitor_mode = false;
    std::vector<double> alpha_divergences;

public:
    AlphaCRF(int W, int H, int M, float alpha);
    virtual ~AlphaCRF();

    MatrixXf inference();
    MatrixXf sequential_inference();
    void keep_track_of_steps();
    void damp_updates(float damp_coeff);
    void compute_exact_marginals();
protected:
    void mfiter_for_proxy_marginals(MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2);
    void cccpiter_for_proxy_marginals(MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2);
    void estimate_proxy_marginals(MatrixXf & approx_Q, MatrixXf & tmp1, MatrixXf & tmp2);
    void proxy_marginals_bf(MatrixXf & approx_Q);
    void weight_pairwise(float coeff);
    double alpha_div(const MatrixXf & approx_Q) const;
    double kl_div(const MatrixXf & approx_Q) const;
};
