#include "densecrf.h"

#ifdef NDEBUG
#define D(x)
#else
#define D(x) do { std::cout << x <<std::endl; } while (0)
#endif

class AlphaCRF: public DenseCRF2D {
protected:

    //factors of the proxy-distribution that we are minimizing for.
    MatrixXf proxy_unary;
    //no pairwise because we store them in pairwise_, as they don't need to be modified.

    // Alpha for which we minimize the alpha-divergence.
    float alpha;

    // Use damping or not, when updating the approximation
    bool use_damping = false;
    float damping_factor = 1e-3;

    // Useful to have to compute the alpha-divergences
    std::vector<float> pairwise_weights;
    std::vector<MatrixXf> pairwise_features;

    // Brute-force the marginals or update them with Alpha-divergences
    bool exact_marginals_mode = false;
    // History of all the alpha divergences that we went through
    bool monitor_mode = false;
    std::vector<double> alpha_divergences;

public:
    AlphaCRF(int W, int H, int M, float alpha);
    virtual ~AlphaCRF();
    void addPairwiseEnergy( const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );

    MatrixXf inference();
    void keep_track_of_steps();
    void damp_updates(float damping_factor);
    void compute_exact_marginals();
protected:
    void mf_for_marginals(MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2);
    void estimate_marginals(MatrixXf & approx_Q, MatrixXf & tmp1, MatrixXf & tmp2);
    void marginals_bf(MatrixXf & approx_Q);
};
