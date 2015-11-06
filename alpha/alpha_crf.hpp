#include "densecrf.h"

class AlphaCRF: public DenseCRF2D {
protected:
    int N_, M_; // Numbers of variables, numbers of labels.

    UnaryEnergy * unary_;
    std::vector<PairwisePotential*> pairwise_;

    float alpha; // Alpha for which we minimize the alpha-divergence.
public:
    AlphaCRF(int W, int H, int M, float alpha);
    virtual ~AlphaCRF();

    MatrixXf inference(int nb_iterations);
};
