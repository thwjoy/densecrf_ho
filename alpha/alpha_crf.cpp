#include "alpha_crf.hpp"
#include "brute_force.hpp"
#include <iostream>
#include <deque>
#include <limits>

void normalize_unaries(MatrixXf & in){
    for (int i=0; i<in.cols(); i++) {
        float col_min = in.col(i).minCoeff();
        in.col(i).array() -= col_min;
    }
}

//////////////////////////////////
// Constructors and Destructors //
//////////////////////////////////

AlphaCRF::AlphaCRF(int W, int H, int M, float alpha) : DenseCRF2D(W, H, M), alpha(alpha) {
}
AlphaCRF::~AlphaCRF(){}

void AlphaCRF::keep_track_of_steps(){
    monitor_mode = true;
};

void AlphaCRF::damp_updates(float damp_coeff){
    use_damping = true;
    damping_factor = damp_coeff;
}

void AlphaCRF::compute_exact_marginals(){
    exact_marginals_mode = true;
}
////////////////////
// Inference Code //
////////////////////

MatrixXf AlphaCRF::inference(){
    D("Starting inference to minimize alpha-divergence.");
    // Q contains our approximation, unary contains the true
    // distribution unary, approx_Q is the meanfield approximation of
    // the proxy-distribution
    MatrixXf Q(M_, N_), unary(M_, N_), marginals(M_, N_), new_Q(M_,N_);
    // tmp1 and tmp2 are matrix to gather intermediary computations
    MatrixXf tmp1(M_, N_), tmp2(M_, N_);

    std::deque<MatrixXf> previous_Q;
    if(!unary_){
        unary.fill(0);
    } else {
        unary = unary_->get();
    }
    D("Initializing the approximating distribution");
    //expAndNormalize( Q, -unary); // Initialization by the unaries
    Q.fill(1/(float)M_); // Initialization to a uniform distribution
    D("Got initial estimates of the distribution");

    previous_Q.push_back(Q);
    bool continue_minimizing_alpha_div = true;
    float Q_change;
    int nb_approximate_distribution = 0;
    while(continue_minimizing_alpha_div){

        if (monitor_mode) {
            double ad = alpha_div(Q, alpha);
            alpha_divergences.push_back(ad);
        }

        D("Constructing proxy distributions");
        // Compute the factors for the approximate distribution
        //Unaries
        MatrixXf true_unary_part = alpha* unary;

        // Condition the matrix properly so that we can obtain the
        // unaries corresponding to the approximation
        if (Q.minCoeff()<=0) {
            Q += std::numeric_limits<float>::epsilon() * MatrixXf::Ones(Q.rows(), Q.cols());
        }
        MatrixXf approx_part = - (1-alpha) * Q.array().log();
        // This needs to be a minus, so that it can be compensated The
        // computation of the probability will sum it, then negate it
        // as part of the energy (so positive contribution) So when we
        // divide by Q^(1-alpha), after the marginalisation, it will
        // cancel out fine

        proxy_unary = true_unary_part + approx_part;
        //// Pairwise term are created when we set up the CRF because they
        //// are going to remain the same
        // WARNING: numerical trick - we normalize the unaries, which
        // shouldn't change anything.  This consist in making the
        // smallest term 0, so that exp(-unary) isn't already way too
        // big.
        normalize_unaries(proxy_unary);
        D("Done constructing the proxy distribution");

        if (exact_marginals_mode) {
            proxy_marginals_bf(marginals);
        } else{
            estimate_proxy_marginals(marginals, tmp1, tmp2);
        }

        D("Estimate the update rule parameters");
        tmp1 = Q.array().pow(1-alpha);
        tmp2 = marginals.cwiseQuotient(tmp1);
        tmp2 = tmp2.array().pow(1/alpha);
        normalize(new_Q, tmp2);

        if(use_damping){
            Q = Q.array().pow(damping_factor) * new_Q.array().pow(1-damping_factor);
        } else {
            Q = new_Q;
        }

        float min_Q_change = std::numeric_limits<float>::max();
        for (std::deque<MatrixXf>::reverse_iterator prev = previous_Q.rbegin(); prev != previous_Q.rend(); prev++) {
            Q_change = (*prev - Q).squaredNorm();
            min_Q_change = min_Q_change < Q_change ? min_Q_change : Q_change;
            continue_minimizing_alpha_div = (min_Q_change > 0.001);
            if(not continue_minimizing_alpha_div){
                break;
            }
        }
        previous_Q.push_back(Q);
        D("Updated our approximation");
        ++nb_approximate_distribution;
    }

    D("Done with alpha-divergence minimization");
    if (monitor_mode) {
        double ad = alpha_div(Q, alpha);
        alpha_divergences.push_back(ad);
    }

    return Q;
}

MatrixXf AlphaCRF::sequential_inference(){
    D("Starting inference to minimize alpha-divergence.");
    // Q contains our approximation, unary contains the true
    // distribution unary, approx_Q is the meanfield approximation of
    // the proxy-distribution
    MatrixXf Q(M_, N_), unary(M_, N_), marginals(M_, N_);
    // tmp1 and tmp2 are matrix to gather intermediary computations
    MatrixXf tmp1(1,N_), tmp2(1,N_);

    unary = unary_->get();

    D("Initializing the approximating distribution");
    //expAndNormalize( Q, -unary); // Initialization by the unaries
    Q.fill(1/(float)M_); // Initialization to a uniform distribution
    D("Got initial estimates of the distribution");

    bool continue_minimizing_alpha_div = true;
    double previous_ad = std::numeric_limits<double>::max();
    double ad;
    while(continue_minimizing_alpha_div){
        ad = alpha_div(Q, alpha);
        for (int var = 0; var < N_; var++) {
            // This needs to be a minus, so that it can be compensated
            // The computation of the probability will sum it, then negate it as part of the energy (so positive contribution)
            // So when we divide by Q^(1-alpha), after the marginalisation, it will cancel out fine
            MatrixXf approx_part = - (1-alpha) * (Q.array().log());
            MatrixXf true_unary_part =  alpha * unary;
            proxy_unary = true_unary_part + approx_part;
            normalize_unaries(proxy_unary);

            proxy_marginals_bf(marginals);

            tmp1 = Q.col(var).array().pow(1-alpha);
            tmp2 = marginals.col(var).cwiseQuotient(tmp1);
            tmp2 = tmp2.array().pow(1/alpha);
            normalize(tmp1, tmp2);
            Q.col(var) = tmp1;

            ad = alpha_div(Q, alpha);
            alpha_divergences.push_back(ad);
        }
        std::cout << "\t\t\t" << ad << '\n';
        continue_minimizing_alpha_div = (previous_ad != ad);
        previous_ad = ad;
    }

    return Q;
}


// Reuse the same tempvariables at all step.
void AlphaCRF::mfiter_for_proxy_marginals(MatrixXf & approx_Q, MatrixXf & tmp1, MatrixXf & tmp2) {
    tmp1 = -proxy_unary;

    for (int i=0; i<pairwise_.size(); i++) {
        pairwise_[i]->apply(tmp2, approx_Q);
        tmp1 -= tmp2;
    }

    expAndNormalize(approx_Q, tmp1);
}

void AlphaCRF::estimate_proxy_marginals(MatrixXf & approx_Q, MatrixXf & tmp1, MatrixXf & tmp2){
    /**
     * approx_Q is a M_ by N_ matrix containing all our marginals that we want to estimate.
     * approx_Q_old .... that contains the previous marginals estimation so that we can estimate convergences.
     * tmp1 and tmp2 are also of the same size, they are temporary matrix, used to perform computations.
     * We pass all of these so that there is no need to reallocate / deallocate.
     */
    D("Starting to estimate the marginals of the distribution");

    // Set the pairwise terms to their weighted version
    weight_pairwise(alpha);

    // Starting value.
    //expAndNormalize(approx_Q, -proxy_unary); // Initialization by the unaries
    approx_Q.fill(1/(float)M_);// Uniform initialization

    std::deque<MatrixXf> previous_Q;
    previous_Q.push_back(approx_Q);


    // Setup the checks for convergence.
    bool continue_estimating_marginals = true;
    float marginal_change;
    int nb_marginal_estimation = 0;

    while(continue_estimating_marginals) {
        // Perform one meanfield iteration to update our approximation
        mfiter_for_proxy_marginals(approx_Q, tmp1, tmp2);
        // If we stopped changing a lot, stop the loop and
        // consider we have some good marginals
        float min_Q_change = std::numeric_limits<float>::max();
        for (std::deque<MatrixXf>::reverse_iterator prev = previous_Q.rbegin(); prev != previous_Q.rend(); ++prev) {
            marginal_change = (*prev - approx_Q).squaredNorm();
            min_Q_change = min_Q_change < marginal_change ? min_Q_change : marginal_change;
            continue_estimating_marginals = (min_Q_change > 0.001);
            if(not continue_estimating_marginals){
                break;
            }
        }
        previous_Q.push_back(approx_Q);
        ++ nb_marginal_estimation;
    }

    // Reset the pairwise terms to their normal version
    weight_pairwise(1/alpha);

    D("Finished MF marginals estimation");
}

void AlphaCRF::proxy_marginals_bf(MatrixXf & approx_Q){
    std::vector<MatrixXf> pairwise_features;
    std::vector<MatrixXf> proxy_weights;
    for (int i = 0; i < pairwise_.size(); i++) {
        pairwise_features.push_back(pairwise_[i]->features());
        proxy_weights.push_back(pairwise_[i]->parameters());
    }
    approx_Q = brute_force_marginals(proxy_unary, pairwise_features, proxy_weights);
}

double AlphaCRF::alpha_div(const MatrixXf & approx_Q, float alpha) const{
    MatrixXf unary = unary_->get();
    std::vector<MatrixXf> pairwise_features;
    std::vector<MatrixXf> pairwise_weights;
    for (int i = 0; i < pairwise_.size(); i++) {
        pairwise_features.push_back(pairwise_[i]->features());
        pairwise_weights.push_back(pairwise_[i]->parameters());
    }
    return compute_alpha_divergence(unary, pairwise_features, pairwise_weights, approx_Q, alpha);
}

void AlphaCRF::weight_pairwise(float coeff){
    // We need to set the pairwise terms with their alpha weightings:
    // - multiply them with Alpha during the iterations
    // - or divide them by Alpha to get back the proper pairwise terms
    for (int i = 0; i < pairwise_.size(); i++) {
        VectorXf parameters = pairwise_[i]->parameters();
        pairwise_[i]->setParameters(coeff * parameters);
    }

}
