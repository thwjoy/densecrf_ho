#include "newton_cccp.hpp"
#include <Eigen/Eigenvalues>

void newton_cccp(VectorXf & state, const VectorXf & cste, float lambda_eig){
    int M_ = cste.size();
    VectorXf kkts(M_ + 1);
    kkts.head(M_) = 2 * lambda_eig * state.head(M_);
    kkts.head(M_) += state.head(M_).array().log().matrix();
    kkts.head(M_) += cste + state(M_) * VectorXf::Ones(M_);
    kkts(M_) = state.head(M_).sum() - 1;
    while (not all_close_to_zero(kkts, 0.001)) {
        // Compute J-1
        VectorXf inv_proba(M_);
        float z_norm = 0;
        for (int l=0; l < M_; l++) {
            inv_proba(l) = state(l) / (1+ 2 * lambda_eig * state(l));
            z_norm += inv_proba(l);
        }
        MatrixXf J1(M_+1, M_+1);
        for (int l=0; l < M_; l++) {
            J1(l,l) = (1-inv_proba(l)/z_norm) * inv_proba(l);
        }
        for (int l1=0; l1 < M_; l1++) {
            for (int l2=l1+1; l2< M_; l2++) {
                float temp = (- inv_proba(l2)/z_norm) * inv_proba(l1);
                J1(l1,l2) = temp;
                J1(l2,l1) = temp;
            }
        }
        for (int l=0; l<M_; l++) {
            J1(M_, l) = inv_proba(l)/z_norm;
            J1(l, M_) = inv_proba(l)/z_norm;
        }
        J1(M_, M_) = -1 / z_norm;

        // Apply J-1 to kkts
        VectorXf step = J1 * (-kkts);

        // Step progressively
        VectorXf new_state = state + step;
        float step_size = 1;
        while(true){
            if(all_positive(new_state.head(M_))){
                break;
            } else {
                step_size = step_size /2;
                new_state = new_state - step_size * step;
            }
        }
        state = new_state;
        kkts.head(M_) = 2 * lambda_eig * state.head(M_);
        kkts.head(M_) += state.head(M_).array().log().matrix();
        kkts.head(M_) += cste + state(M_) * VectorXf::Ones(M_);
        kkts(M_) = state.head(M_).sum() - 1;
    }

}


float pick_lambda_eig_to_concave(MatrixXf const & lbl_compatibility){
    // We make the assumption that the label compatibility is symmetric,
    // which other places in the code do.
    VectorXf eigs = lbl_compatibility.selfadjointView<Upper>().eigenvalues();
    float lambda_eig = eigs.maxCoeff();
    lambda_eig = lambda_eig > 0? lambda_eig : 0;
    return lambda_eig;
}
