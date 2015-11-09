#include "densecrf.h"

class AlphaCRF: public DenseCRF2D {
protected:

    //factors of the proxy-distribution that we are minimizing for.
    MatrixXf proxy_unary;
    //no pairwise because we store them in pairwise_, as they don't need to be modified.

    // Alpha for which we minimize the alpha-divergence.
    float alpha;

    // Hyperparameters on how to run the algorithm.
    int nb_mf_marg;

public:
    AlphaCRF(int W, int H, int M, float alpha, int nb_mf_marg=3);
    virtual ~AlphaCRF();
    void addPairwiseEnergy( const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );

    MatrixXf inference(int nb_iterations);

protected:
    void mf_for_marginals(MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2);

};
