/*
    Copyright (c) 2013, Philipp Krähenbühl

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
        DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
        (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
        LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
        ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "densecrf.h"
#include "eigen_utils.hpp"
#include "newton_cccp.hpp"
#include "qp.hpp"
#include "permutohedral.h"
#include "util.h"
#include "pairwise.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>

#define BRUTE_FORCE false	// brute-force subgraient computation, used in lp_prox and energy computations
#define VERBOSE false	    // print intermediate energy values and timings, used in lp_prox

#define DCNEG_FASTAPPROX false
/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////
DenseCRF::DenseCRF(int N, int M) : N_(N), M_(M), unary_(0) {
}
DenseCRF::~DenseCRF() {
    if (unary_)
        delete unary_;
    for( unsigned int i=0; i<pairwise_.size(); i++ )
        delete pairwise_[i];
    for( unsigned int i=0; i<no_norm_pairwise_.size(); i++ )
        delete no_norm_pairwise_[i];
}
DenseCRF2D::DenseCRF2D(int W, int H, int M) : DenseCRF(W*H,M), W_(W), H_(H) {
}
DenseCRF2D::~DenseCRF2D() {
}
/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::setPairwisePottsWeight(float ratio, const MatrixXf & Q) {
	VectorXs l = currentMap(Q);
	double u_energy = unaryEnergy(l).sum();
    double p_energy = pairwise_energy_true(l).sum();
	double p_weight = ratio*u_energy/p_energy;
	std::cout << "#p_weight: " << p_weight << std::endl;
	for( unsigned int k=0; k<pairwise_.size(); k++ ) {
		pairwise_[k]->setParameters(pairwise_[k]->parameters() * p_weight);
		no_norm_pairwise_[k]->setParameters(no_norm_pairwise_[k]->parameters() * p_weight);
	}
}
void DenseCRF::addPairwiseEnergy (const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type, NormalizationType normalization_type) {
    assert( features.cols() == N_ );
    addPairwiseEnergy( new PairwisePotential( features, function, kernel_type, normalization_type ) );
}
void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential ){
    pairwise_.push_back( potential );
    // no-norm-pairwise
    no_norm_pairwise_.push_back( new PairwisePotential(
            potential->features(),
            new PottsCompatibility(potential->parameters()(0)),
            potential->ktype(),
            NO_NORMALIZATION) );
}
void DenseCRF2D::addPairwiseGaussian ( float sx, float sy, LabelCompatibility * function, KernelType kernel_type, NormalizationType normalization_type ) {
    MatrixXf feature( 2, N_ );
    for( int j=0; j<H_; j++ )
        for( int i=0; i<W_; i++ ){
            feature(0,j*W_+i) = i / sx;
            feature(1,j*W_+i) = j / sy;
        }
    addPairwiseEnergy( feature, function, kernel_type, normalization_type );
}
void DenseCRF2D::addPairwiseBilateral ( float sx, float sy, float sr, float sg, float sb, const unsigned char* im, LabelCompatibility * function, KernelType kernel_type, NormalizationType normalization_type ) {
    MatrixXf feature( 5, N_ );
    for( int j=0; j<H_; j++ )
        for( int i=0; i<W_; i++ ){
            feature(0,j*W_+i) = i / sx;
            feature(1,j*W_+i) = j / sy;
            feature(2,j*W_+i) = im[(i+j*W_)*3+0] / sr;
            feature(3,j*W_+i) = im[(i+j*W_)*3+1] / sg;
            feature(4,j*W_+i) = im[(i+j*W_)*3+2] / sb;
        }
    addPairwiseEnergy( feature, function, kernel_type, normalization_type );
}
//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////
void DenseCRF::setUnaryEnergy ( UnaryEnergy * unary ) {
    if( unary_ ) delete unary_;
    unary_ = unary;
}
void DenseCRF::setUnaryEnergy( const MatrixXf & unary ) {
    setUnaryEnergy( new ConstUnaryEnergy( unary ) );
}
void  DenseCRF::setUnaryEnergy( const MatrixXf & L, const MatrixXf & f ) {
    setUnaryEnergy( new LogisticUnaryEnergy( L, f ) );
}

UnaryEnergy* DenseCRF::getUnaryEnergy() {
    return unary_;
}
///////////////////////
/////  Inference  /////
///////////////////////
LP_inf_params::LP_inf_params(float prox_reg_const, float dual_gap_tol, float prox_energy_tol, 
        int prox_max_iter, int fw_max_iter, 
        int qp_max_iter, float qp_tol, float qp_const, 
        bool best_int, bool accel_prox, 
        int work_set_size, int approx_fw_iter,
        float less_confident_percent, float confidence_tol):
        prox_reg_const(prox_reg_const),	dual_gap_tol(dual_gap_tol), prox_energy_tol(prox_energy_tol), 
        prox_max_iter(prox_max_iter), fw_max_iter(fw_max_iter), 
        qp_max_iter(qp_max_iter), qp_tol(qp_tol), qp_const(qp_const), 
        best_int(best_int), accel_prox(accel_prox),
        work_set_size(work_set_size), approx_fw_iter(approx_fw_iter),
        less_confident_percent(less_confident_percent), confidence_tol(confidence_tol) {}

LP_inf_params::LP_inf_params() {
	prox_reg_const = 1e-2;	
	dual_gap_tol = 1e3;		
	prox_energy_tol = 1e3;		
	prox_max_iter = 10;		
	fw_max_iter = 10;		
	qp_max_iter = 1000;		
	qp_tol = 1e3;			
	qp_const = 1e-16;			
    best_int = true;
    accel_prox = true;
    work_set_size = 10;
    approx_fw_iter = 10;
    less_confident_percent = 0;  // don't check for less confident pixels
    confidence_tol = 0.95;
}

LP_inf_params::LP_inf_params(const LP_inf_params& params) {
	prox_reg_const = params.prox_reg_const;
	dual_gap_tol = params.dual_gap_tol;		
	prox_energy_tol = params.prox_energy_tol;		
	prox_max_iter = params.prox_max_iter;		
	fw_max_iter = params.fw_max_iter;		
	qp_max_iter = params.qp_max_iter;		
	qp_tol = params.qp_tol;			
	qp_const = params.qp_const;			
    best_int = params.best_int;
    accel_prox = params.accel_prox;
    work_set_size = params.work_set_size;
    approx_fw_iter = params.approx_fw_iter;
    less_confident_percent = params.less_confident_percent;
    confidence_tol = params.confidence_tol;
}

MatrixXf DenseCRF::unary_init() const {
    MatrixXf Q;
    expAndNormalize(Q, -unary_->get());
    return Q;
}

MatrixXf DenseCRF::uniform_init() const{
    MatrixXf Q = (1.0 / M_) * MatrixXf::Ones(M_, N_);
    return Q;
}

void normalize(MatrixXf & out, const MatrixXf & in){
    out.resize(in.rows(), in.cols());
    VectorXf norm_constants = in.colwise().sum();
    out = in.array().rowwise() / norm_constants.array().transpose();
}

void expAndNormalize ( MatrixXf & out, const MatrixXf & in ) {
    out.resize( in.rows(), in.cols() );
    for( int i=0; i<out.cols(); i++ ){
        VectorXf b = in.col(i);
        b.array() -= b.maxCoeff();
        b = b.array().exp();
        out.col(i) = b / b.array().sum();
    }
}
void sumAndNormalize( MatrixXf & out, const MatrixXf & in, const MatrixXf & Q ) {
    out.resize( in.rows(), in.cols() );
    for( int i=0; i<in.cols(); i++ ){
        VectorXf b = in.col(i);
        VectorXf q = Q.col(i);
        out.col(i) = b.array().sum()*q - b;
    }
}

MatrixXf DenseCRF::inference ( const MatrixXf & init, int n_iterations ) const {
    MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2;
    unary.fill(0);
    if( unary_ ){
        unary = unary_->get();
    }
    Q = init;

    for( int it=0; it<n_iterations; it++ ) {
        tmp1 = -unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp2, Q );
            tmp1 -= tmp2;
        }
        expAndNormalize( Q, tmp1 );
    }
    return Q;
}

std::vector<perf_measure> DenseCRF::tracing_inference(MatrixXf & init, double time_limit) const {
    MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2, old_Q(M_, N_);
    float old_kl, kl;

    clock_t start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;
    
    MatrixXf best_Q = Q;
    double best_int_energy = std::numeric_limits<double>::max();
    double int_energy = 0;

    unary.fill(0);
    if( unary_ ){
        unary = unary_->get();
    }

    Q = init;

    if (compute_kl) {
        old_kl = 0;
        kl = klDivergence(Q);
    }

    bool keep_inferring = true;
    old_Q = Q;
    int count = 0;
    while(keep_inferring or time_limit != 0) {
        start = clock();
        old_kl = kl;
        tmp1 = -unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp2, Q );
            tmp1 -= tmp2;
        }
        expAndNormalize( Q, tmp1 );


        if (compute_kl) {
            kl = klDivergence(Q);
            float kl_change = old_kl - kl;
            keep_inferring = (kl_change > 0.001);
            old_kl = kl;
        } else {
            float Q_change = (old_Q - Q).squaredNorm();
            keep_inferring = (Q_change > 0.001);
        }
        old_Q = Q;

        end = clock();
        perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        int_energy = assignment_energy_true(currentMap(Q));
        if (best_int_energy > int_energy) {
            best_int_energy = int_energy;
            best_Q = Q;
        }
        perf_energy = best_int_energy;
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;
        if (time_limit != 0 and total_time>time_limit) {
            break;
        }

        count++;
    }
    init = Q;
    return perfs;
}

MatrixXf DenseCRF::inference (const MatrixXf & init) const {
    MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2, old_Q(M_, N_);
    float old_kl, kl;
    unary.fill(0);
    if( unary_ ){
        unary = unary_->get();
    }

    Q = init;

    if (compute_kl) {
        old_kl = 0;
        kl = klDivergence(Q);
    }

    bool keep_inferring = true;
    old_Q = Q;
    int count = 0;
    while(keep_inferring) {
        old_kl = kl;
        tmp1 = -unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp2, Q );
            tmp1 -= tmp2;
        }
        expAndNormalize( Q, tmp1 );

        if (compute_kl) {
            kl = klDivergence(Q);
            float kl_change = old_kl - kl;
            keep_inferring = (kl_change > 0.001);
            old_kl = kl;
        } else {
            float Q_change = (old_Q - Q).squaredNorm();
            keep_inferring = (Q_change > 0.001);
        }
        old_Q = Q;
        count++;
    }
    return Q;
}

MatrixXf DenseCRF::qp_inference(const MatrixXf & init) const {
    MatrixXf Q(M_, N_), unary(M_, N_), diag_dom(M_,N_), tmp(M_,N_), grad(M_, N_),
        desc(M_,N_), sx(M_,N_), psisx(M_, N_);
    MatrixP temp_dot(M_,N_);
    double optimal_step_size = 0;
    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }
    // Initialize state to the unaries
    // Warning: We don't get exactly the same optimum depending on the initialisation
    // expAndNormalize(Q, -unary);
    Q = init;

    // Build proxy unaries for the added terms
    // Compute the dominant diagonal

    // Note: All the terms in the pairwise matrix are negatives
    // so to get the sum of the abs value, you need to get the
    // product with the matrix full of -1.
    diag_dom.fill(0);
    MatrixXf full_ones = -MatrixXf::Ones(M_, N_);
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, full_ones);
        diag_dom += tmp;
    }
    diag_dom += 0.0001 * MatrixXf::Ones(M_, N_);
    // This is a caution to make sure that the matrix is well
    // diagonally dominant and therefore convex. Otherwise, due to
    // floating point errors, we can have issues.
    // This is triggered easily when we initialise with the uniform distribution,
    // then the results of the matrix multiplication is exactly the sum along the columns.

    // Update the proxy_unaries
    unary = unary - diag_dom;

    // Compute the value of the energy
    double old_energy = std::numeric_limits<double>::max();
    double energy;

    grad = unary;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q);
        grad += 2 *tmp;
    }
    grad += 2 * diag_dom.cwiseProduct(Q);


    energy = compute_LR_QP_value(Q, diag_dom);
    while( (old_energy - energy) > 100){
        old_energy = energy;

        // Get a Descent direction by minimising < \nabla E, s >
        descent_direction(desc, grad);
        // Solve for the best step size. The best step size is
        // - \frac{\theta'^T(s-x) + 2 x^T \psi (s-x)}{ 2 * (s-x)^T \psi (s-x) }
        sx = desc - Q;
        psisx.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx);
            psisx += tmp;
        }
        psisx += diag_dom.cwiseProduct(sx);

        double num =  2 * dotProduct(Q, psisx, temp_dot) + dotProduct(unary, sx, temp_dot);
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx, psisx, temp_dot);
        // Denom should be positive, otherwise our choice of psi was not convex enough.

        optimal_step_size = - num / (2 * denom);
        if (optimal_step_size > 1) {
            // Respect the bounds.
            optimal_step_size = 1;
        }
        if (optimal_step_size < 0) {
            // Stay between the current step and the optimal.
            // Theoretically shouldn't happen but we accumulate
            // floating point errors when we compute the polynomial
            // coefficients.
            optimal_step_size = 0;
        }
        if (denom == 0) {
            // This means that the conditional gradient is the same
            // than the current step and we have converged.
            optimal_step_size = 0;
        }
        // Take a step
        Q += optimal_step_size * sx;
        if (not valid_probability(Q)) {
            std::cout << "Bad proba" << '\n';
        }
        // Compute the gradient at the new estimates.
        grad += 2* optimal_step_size * psisx;
        //energy = compute_LR_QP_value(Q, diag_dom);
        //alt_energy = dotProduct(Q, unary, temp_dot) + 0.5*dotProduct(Q, grad - unary, temp_dot);
        energy = 0.5 * dotProduct(Q, grad + unary, temp_dot);
    }
    return Q;
}

MatrixXf DenseCRF::qp_inference(const MatrixXf & init, int nb_iterations) const {
    MatrixXf Q(M_, N_), unary(M_, N_), diag_dom(M_,N_), tmp(M_,N_), grad(M_, N_),
        desc(M_,N_), sx(M_,N_), psisx(M_, N_);
    MatrixP temp_dot(M_,N_);
    double optimal_step_size = 0;
    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }
    // Initialize state to the unaries
    // Warning: We don't get exactly the same optimum depending on the initialisation
    // expAndNormalize(Q, -unary);
    Q = init;

    // Build proxy unaries for the added terms
    // Compute the dominant diagonal

    // Note: All the terms in the pairwise matrix are negatives
    // so to get the sum of the abs value, you need to get the
    // product with the matrix full of -1.
    diag_dom.fill(0);
    MatrixXf full_ones = -MatrixXf::Ones(M_, N_);
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, full_ones);
        diag_dom += tmp;
    }
    diag_dom += 0.0001 * MatrixXf::Ones(M_, N_);
    // This is a caution to make sure that the matrix is well
    // diagonally dominant and therefore convex. Otherwise, due to
    // floating point errors, we can have issues.
    // This is triggered easily when we initialise with the uniform distribution,
    // then the results of the matrix multiplication is exactly the sum along the columns.

    // Update the proxy_unaries
    unary = unary - diag_dom;

    // Compute the value of the energy
    double old_energy = std::numeric_limits<double>::max();
    double energy;

    grad = unary;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q);
        grad += 2 *tmp;
    }
    grad += 2 * diag_dom.cwiseProduct(Q);


    energy = compute_LR_QP_value(Q, diag_dom);
    for(int it = 0; it < nb_iterations; ++it){
        old_energy = energy;

        // Get a Descent direction by minimising < \nabla E, s >
        descent_direction(desc, grad);
        // Solve for the best step size. The best step size is
        // - \frac{\theta'^T(s-x) + 2 x^T \psi (s-x)}{ 2 * (s-x)^T \psi (s-x) }
        sx = desc - Q;
        psisx.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx);
            psisx += tmp;
        }
        psisx += diag_dom.cwiseProduct(sx);

        double num =  2 * dotProduct(Q, psisx, temp_dot) + dotProduct(unary, sx, temp_dot);
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx, psisx, temp_dot);
        // Denom should be positive, otherwise our choice of psi was not convex enough.

        optimal_step_size = - num / (2 * denom);
        if (optimal_step_size > 1) {
            // Respect the bounds.
            optimal_step_size = 1;
        }
        if (optimal_step_size < 0) {
            // Stay between the current step and the optimal.
            // Theoretically shouldn't happen but we accumulate
            // floating point errors when we compute the polynomial
            // coefficients.
            optimal_step_size = 0;
        }
        if (denom == 0) {
            // This means that the conditional gradient is the same
            // than the current step and we have converged.
            optimal_step_size = 0;
        }
        // Take a step
        Q += optimal_step_size * sx;
        if (not valid_probability(Q)) {
            std::cout << "Bad proba" << '\n';
        }
        // Compute the gradient at the new estimates.
        grad += 2* optimal_step_size * psisx;
        //energy = compute_LR_QP_value(Q, diag_dom);
        //alt_energy = dotProduct(Q, unary, temp_dot) + 0.5*dotProduct(Q, grad - unary, temp_dot);
        //energy = 0.5 * dotProduct(Q, grad + unary, temp_dot);
    }
    return Q;
}


std::vector<perf_measure> DenseCRF::tracing_qp_inference(MatrixXf & init, double time_limit) const {
    MatrixXf Q(M_, N_), unary(M_, N_), diag_dom(M_,N_), tmp(M_,N_), grad(M_, N_),
        desc(M_,N_), sx(M_,N_), psisx(M_, N_);
    MatrixP temp_dot(M_,N_);
    double energy, old_energy;


    clock_t start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;

    double optimal_step_size = 0;
    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }
    // Initialize state to the unaries
    // Warning: We don't get exactly the same optimum depending on the initialisation
    // expAndNormalize(Q, -unary);
    Q = init;
    MatrixXf best_Q = Q;
    double best_int_energy = std::numeric_limits<double>::max();
    double int_energy = 0;

    // Build proxy unaries for the added terms
    // Compute the dominant diagonal

    start = clock();

    {
        // Note: All the terms in the pairwise matrix are negatives
        // so to get the sum of the abs value, you need to get the
        // product with the matrix full of -1.
        diag_dom.fill(0);
        MatrixXf full_ones = -MatrixXf::Ones(M_, N_);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, full_ones);
            diag_dom += tmp;
        }
        diag_dom += 0.0001 * MatrixXf::Ones(M_, N_);
        // This is a caution to make sure that the matrix is well
        // diagonally dominant and therefore convex. Otherwise, due to
        // floating point errors, we can have issues.
        // This is triggered easily when we initialise with the uniform distribution,
        // then the results of the matrix multiplication is exactly the sum along the columns.

        // Update the proxy_unaries
        unary = unary - diag_dom;

        // Compute the value of the energy
        old_energy = std::numeric_limits<double>::max();

        grad = unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            grad += 2 *tmp;
        }
        grad += 2 * diag_dom.cwiseProduct(Q);


        energy = compute_LR_QP_value(Q, diag_dom);
    } // This is some necessary setup for the QP inference so this need to be accounted.

    end = clock();
    perf_timing = (double(end-start)/CLOCKS_PER_SEC);
    int_energy = assignment_energy_true(currentMap(Q));
    if (best_int_energy > int_energy) {
        best_int_energy = int_energy;
        best_Q = Q;
    }
    perf_energy = best_int_energy;
    latest_perf = std::make_pair(perf_timing, perf_energy);
    perfs.push_back(latest_perf);
    total_time += perf_timing;


    while( (old_energy - energy) > 100 or time_limit != 0){
        start = clock();
        old_energy = energy;

        // Get a Descent direction by minimising < \nabla E, s >
        descent_direction(desc, grad);
        // Solve for the best step size. The best step size is
        // - \frac{\theta'^T(s-x) + 2 x^T \psi (s-x)}{ 2 * (s-x)^T \psi (s-x) }
        sx = desc - Q;
        psisx.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx);
            psisx += tmp;
        }
        psisx += diag_dom.cwiseProduct(sx);

        double num =  2 * dotProduct(Q, psisx, temp_dot) + dotProduct(unary, sx, temp_dot);
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx, psisx, temp_dot);
        // Denom should be positive, otherwise our choice of psi was not convex enough.

        optimal_step_size = - num / (2 * denom);
        if (optimal_step_size > 1) {
            // Respect the bounds.
            optimal_step_size = 1;
        }
        if (optimal_step_size < 0) {
            // Stay between the current step and the optimal.
            // Theoretically shouldn't happen but we accumulate
            // floating point errors when we compute the polynomial
            // coefficients.
            optimal_step_size = 0;
        }
        if (denom == 0) {
            // This means that the conditional gradient is the same
            // than the current step and we have converged.
            optimal_step_size = 0;
        }
        // Take a step
        Q += optimal_step_size * sx;

        if (not valid_probability(Q)) {
            std::cout << "Bad proba" << '\n';
        }
        // Compute the gradient at the new estimates.
        grad += 2* optimal_step_size * psisx;
        //energy = compute_LR_QP_value(Q, diag_dom);
        //alt_energy = dotProduct(Q, unary, temp_dot) + 0.5*dotProduct(Q, grad - unary, temp_dot);
        energy = 0.5 * dotProduct(Q, grad + unary, temp_dot);

        // performance measurement
        end = clock();
        perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        int_energy = assignment_energy_true(currentMap(Q));
        if (best_int_energy > int_energy) {
            best_int_energy = int_energy;
            best_Q = Q;
        }
        perf_energy = best_int_energy;
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;
        if (time_limit != 0 and total_time>time_limit) {
            break;
        }
    }
    init = Q;
    return perfs;
}




void kkt_solver(const VectorXf & lin_part, const MatrixXf & inv_KKT, VectorXf & out){
    int M = lin_part.size();
    VectorXf state(M + 1);
    VectorXf target(M + 1);
    target.head(M) = -lin_part;
    target(M) = 1;
    state = inv_KKT * target;
    out = state.head(M);
}



MatrixXf DenseCRF::concave_qp_cccp_inference(const MatrixXf & init) const {
    MatrixXf Q(M_, N_), unary(M_, N_), tmp(M_, N_),
        outer_grad(M_,N_), psis(M_, N_);
    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    Q = init;
    // Compute the value of the energy
    double old_energy;
    double energy = compute_energy(Q);
    int outer_rounds = 0;
    double identity_coefficient = 0;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        identity_coefficient -= pairwise_[k]->parameters()(0);
    }

    MatrixXf inv_KKT(M_+1, M_+1);
    for (int i=0; i < M_; i++) {
        for (int j=0; j < M_; j++) {
            inv_KKT(i,j) = 1 / (M_ * 2*identity_coefficient);
            if (i==j) {
                inv_KKT(i,j) -= 1/ (2*identity_coefficient);
            }
        }
        inv_KKT(M_, i) = 1.0/M_;
        inv_KKT(i, M_) = 1.0/M_;
    }
    inv_KKT(M_,M_) = 2*identity_coefficient / M_;


// MatrixXf KKT(M_+1, M_+1);
// KKT.fill(0);
// for (int i=0; i < M_; i++) {
//     KKT(i,i) = 2 * identity_coefficient;
//     KKT(M_, i) = 1;
//     KKT(i, M_) = 1;
// }
// std::cout << inv_KKT*KKT << '\n';


    VectorXf new_Q(M_);
    float old_score, score;
    VectorXf grad(M_), cond_grad(M_), desc(M_);
    int best_coord;
    float num, denom, optimal_step_size;
    bool has_converged;
    int nb_iter;
    do {
        // New value of the linearization point.
        old_energy = energy;
        psis.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            psis += tmp;
        }
        outer_grad = unary + 2 * psis;
        cond_grad.fill(0);
        for (int var = 0; var < N_; ++var) {
#if DCNEG_FASTAPPROX
            kkt_solver(outer_grad.col(var), inv_KKT, new_Q);
            clamp_and_normalize(new_Q);
#else
            kkt_solver(outer_grad.col(var), inv_KKT, new_Q);
            score = outer_grad.col(var).dot(new_Q) + identity_coefficient * new_Q.squaredNorm();
            if(not all_positive(new_Q)){
                // Our KKT conditions didn't get us the correct results.
                // Let's Frank-wolfe it
                clamp_and_normalize(new_Q);
                // Get an initial valid point.
                score = outer_grad.col(var).dot(new_Q) + identity_coefficient * new_Q.squaredNorm();
                // Get a valid score.
                cond_grad.fill(0);
                has_converged = false;
                nb_iter = 0;
                do{
                    old_score = score;
                    cond_grad.fill(0);
                    grad = outer_grad.col(var) + 2 * identity_coefficient * new_Q;
                    grad.minCoeff(&best_coord);
                    cond_grad(best_coord) = 1;
                    desc = cond_grad - new_Q;
                    if (desc.squaredNorm()==0) {
                        break;
                    }
                    num = grad.dot(desc);
                    if (num > 0) {
                        // Commented out code to identify bugs with this computation
                        if (num > 1e-6) {
                            std::cout << "Shouldn't happen." << '\n';
                            std::cout << "Cond score: " << grad.dot(cond_grad) << '\n';
                            std::cout << "Point score: " << grad.dot(new_Q) << '\n';
                            std::cout << num << '\n';
                            std::cout  << '\n';
                        }
                        num = 0;
                    }
                    denom = -identity_coefficient * desc.squaredNorm();
                    optimal_step_size = - num / (2 * denom);
                    if(optimal_step_size > 1){
                        optimal_step_size = 1; // Would get outta bounds
                    }
                    new_Q += optimal_step_size*desc;
                    score = outer_grad.col(var).dot(new_Q) + identity_coefficient * new_Q.squaredNorm();
                    if (old_score - score < 1e-2) {
                        has_converged = true;
                    }
                    nb_iter++;
                } while(not has_converged);
            }
#endif
            Q.col(var) = new_Q;
        }
        //valid_probability_debug(Q);
        // Compute our current value of the energy;
        energy = compute_energy(Q);
        outer_rounds++;
    } while ( (old_energy -energy) > 100 && outer_rounds < 100);
    return Q;
}

std::vector<perf_measure> DenseCRF::tracing_concave_qp_cccp_inference(MatrixXf & init, double time_limit) const {
    MatrixXf Q(M_, N_), unary(M_, N_), tmp(M_, N_),
        outer_grad(M_,N_), psis(M_, N_);

    clock_t start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;

    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    Q = init;
    // Compute the value of the energy
    double old_energy;
    double energy = compute_energy(Q);
    int outer_rounds = 0;
    double identity_coefficient = 0;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        identity_coefficient -= pairwise_[k]->parameters()(0);
    }

    MatrixXf inv_KKT(M_+1, M_+1);
    for (int i=0; i < M_; i++) {
        for (int j=0; j < M_; j++) {
            inv_KKT(i,j) = 1 / (M_ * 2*identity_coefficient);
            if (i==j) {
                inv_KKT(i,j) -= 1/ (2*identity_coefficient);
            }
        }
        inv_KKT(M_, i) = 1.0/M_;
        inv_KKT(i, M_) = 1.0/M_;
    }
    inv_KKT(M_,M_) = 2*identity_coefficient / M_;

    VectorXf new_Q(M_);
    float old_score, score;
    VectorXf grad(M_), cond_grad(M_), desc(M_);
    int best_coord;
    float num, denom, optimal_step_size;
    bool has_converged;
    int nb_iter;
    MatrixXf best_Q = Q;
    double best_int_energy = std::numeric_limits<double>::max();
    double int_energy = 0;
    do {

        start = clock();
        // New value of the linearization point.
        old_energy = energy;
        psis.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            psis += tmp;
        }
        outer_grad = unary + 2 * psis;
        cond_grad.fill(0);
        for (int var = 0; var < N_; ++var) {
#if DCNEG_FASTAPPROX
            kkt_solver(outer_grad.col(var), inv_KKT, new_Q);
            clamp_and_normalize(new_Q);
#else
            kkt_solver(outer_grad.col(var), inv_KKT, new_Q);
            score = outer_grad.col(var).dot(new_Q) + identity_coefficient * new_Q.squaredNorm();
            if(not all_positive(new_Q)){
                // Our KKT conditions didn't get us the correct results.
                // Let's Frank-wolfe it
                clamp_and_normalize(new_Q);
                // Get an initial valid point.
                score = outer_grad.col(var).dot(new_Q) + identity_coefficient * new_Q.squaredNorm();
                // Get a valid score.
                cond_grad.fill(0);
                has_converged = false;
                nb_iter = 0;
                do{
                    old_score = score;
                    cond_grad.fill(0);
                    grad = outer_grad.col(var) + 2 * identity_coefficient * new_Q;
                    grad.minCoeff(&best_coord);
                    cond_grad(best_coord) = 1;
                    desc = cond_grad - new_Q;
                    if (desc.squaredNorm()==0) {
                        break;
                    }
                    num = grad.dot(desc);
                    if (num > 0) {
                        // Commented out code to identify bugs with this computation
                        // if (num > 1e-6) {
                        //     std::cout << "Shouldn't happen." << '\n';
                        //     std::cout << "Cond score: " << grad.dot(cond_grad) << '\n';
                        //     std::cout << "Point score: " << grad.dot(new_Q) << '\n';
                        //     std::cout << num << '\n';
                        //     std::cout  << '\n';
                        // }
                        num = 0;
                    }
                    denom = -identity_coefficient * desc.squaredNorm();
                    optimal_step_size = - num / (2 * denom);
                    if(optimal_step_size > 1){
                        optimal_step_size = 1; // Would get outta bounds
                    }
                    new_Q += optimal_step_size*desc;
                    score = outer_grad.col(var).dot(new_Q) + identity_coefficient * new_Q.squaredNorm();
                    if (old_score - score < 1e-2) {
                        has_converged = true;
                    }
                    nb_iter++;
                } while(not has_converged);
            }
#endif
            Q.col(var) = new_Q;
        }
        //valid_probability_debug(Q);
        // Compute our current value of the energy;
        end = clock();
        perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        int_energy = assignment_energy_true(currentMap(Q));
        if (best_int_energy > int_energy) {
            best_int_energy = int_energy;
            best_Q = Q;
        }
        //std::cout << "#DC-neg: " << int_energy << ", " << best_int_energy << std::endl;
        perf_energy = best_int_energy;
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;
        if (time_limit != 0 and total_time>time_limit) {
            break;
        }


        energy = compute_energy(Q);
        outer_rounds++;
    } while (( (old_energy -energy) > 100 && outer_rounds < 100) or time_limit != 0);
    init = Q;
    return perfs;
}


MatrixXf DenseCRF::qp_cccp_inference(const MatrixXf & init) const {
    MatrixXf Q(M_, N_), Q_old(M_,N_), grad(M_,N_), unary(M_, N_), tmp(M_, N_),
        desc(M_, N_), sx(M_, N_), psis(M_, N_), diag_dom(M_,N_);
    MatrixP temp_dot(M_,N_);
    // Compute the smallest eigenvalues, that we need to make bigger
    // than 0, to ensure that the problem is convex.
    diag_dom.fill(0);
    MatrixXf full_ones = -MatrixXf::Ones(M_, N_);
    diag_dom.fill(0);
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, full_ones);
        diag_dom += tmp;
    }
    diag_dom += 0.0001 * MatrixXf::Ones(M_, N_);

    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    Q = init;
    // Compute the value of the energy
    double old_energy;
    double energy = compute_energy(Q);
    int outer_rounds = 0;
    do {
        // New value of the linearization point.
        old_energy = energy;
        Q_old = Q;

        double convex_energy = energy + dotProduct(Q, diag_dom.cwiseProduct(2*Q_old - Q), temp_dot);
        double old_convex_energy;
        int convex_rounds = 0;

        double optimal_step_size = 0;

        psis.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            psis += tmp;
        }
        grad = unary + 2 * psis + 2 * diag_dom.cwiseProduct(Q_old - Q);
        do {
            old_convex_energy = convex_energy;

            // Get a Descent direction by minimising < \nabla E, s >
            descent_direction(desc, grad);

            // Solve for the best step size of the convex problem. It
            // is - frac{\phi^T(s-x) + 2 x^T \psi (s-x) + 2(x-x_old)^T
            // d (s-x)}{2 (s-x) (\psi+d) (s-x)}
            sx = desc - Q;

            psis.fill(0);
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
                pairwise_[k]->apply(tmp, sx);
                psis += tmp;
            }

            double num = dotProduct(sx, unary + 2*diag_dom.cwiseProduct(Q-Q_old), temp_dot) +
                2 * dotProduct(Q, psis, temp_dot);
            assert(num<=0); // This is negative if desc is really the good minimizer

            double denom = dotProduct(desc, psis + diag_dom.cwiseProduct(desc), temp_dot);
            assert(denom>0); // This is positive if we did our decomposition correctly

            double cst = dotProduct(Q, 0.5 * (grad + unary) - diag_dom.cwiseProduct(Q_old),temp_dot);

            optimal_step_size = - num/ (2 *denom);

            if (optimal_step_size > 1) {
                optimal_step_size = 1;
            } else if( optimal_step_size < 0){
                optimal_step_size = 0;
            }

            Q += optimal_step_size * sx;
            // Compute gradient of the convex problem at the new position
            grad += 2 * optimal_step_size * (psis-diag_dom.cwiseProduct(sx));

            // std::cout << "Coefficients: "<< denom << '\t' << num << '\t' << cst << '\n';
            convex_energy = pow(optimal_step_size, 2) * denom + optimal_step_size * num + cst;

            // energy = compute_energy(Q);
            convex_rounds++;
        } while ( (old_convex_energy - convex_energy) > 100 && convex_rounds<10 && optimal_step_size != 0);
        // We are now (almost) at a minimum of the convexified problem, so we
        // stop solving the convex problem and get a new convex approximation.


        //Check that the reduction in dotProduct actually corresponds to a decrease in runtime.


        // Compute our current value of the energy;
        // energy = compute_energy(Q);
        energy = dotProduct(Q, 0.5 * (grad + unary) - diag_dom.cwiseProduct(Q_old - Q), temp_dot);
        outer_rounds++;
    } while ( (old_energy -energy) > 100 && outer_rounds < 20);
    return Q;
}

std::vector<perf_measure> DenseCRF::tracing_qp_cccp_inference(MatrixXf & init, double time_limit) const {
    MatrixXf Q(M_, N_), Q_old(M_,N_), grad(M_,N_), unary(M_, N_), tmp(M_, N_),
        desc(M_, N_), sx(M_, N_), psis(M_, N_), diag_dom(M_,N_);
    MatrixP temp_dot(M_,N_);

    clock_t start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;

    Q = init;
    MatrixXf best_Q = Q;
    double best_int_energy = std::numeric_limits<double>::max();
    double int_energy = 0;
    start = clock();
    {
        // Compute the smallest eigenvalues, that we need to make bigger
        // than 0, to ensure that the problem is convex.
        diag_dom.fill(0);
        MatrixXf full_ones = -MatrixXf::Ones(M_, N_);
        diag_dom.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, full_ones);
            diag_dom += tmp;
        }
        diag_dom += 0.0001 * MatrixXf::Ones(M_, N_);
    }
    end = clock();
    perf_timing = (double(end-start)/CLOCKS_PER_SEC);
    int_energy = assignment_energy_true(currentMap(Q));
    if (best_int_energy > int_energy) {
        best_int_energy = int_energy;
        best_Q = Q;
    }
    perf_energy = best_int_energy;
    latest_perf = std::make_pair(perf_timing, perf_energy);
    perfs.push_back(latest_perf);
    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    // Compute the value of the energy
    double old_energy;
    double energy = compute_energy(Q);
    int outer_rounds = 0;
    do {
        start = clock();
        // New value of the linearization point.
        old_energy = energy;
        Q_old = Q;

        double convex_energy = energy + dotProduct(Q, diag_dom.cwiseProduct(2*Q_old - Q), temp_dot);
        double old_convex_energy;
        int convex_rounds = 0;

        double optimal_step_size = 0;

        psis.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            psis += tmp;
        }
        grad = unary + 2 * psis + 2 * diag_dom.cwiseProduct(Q_old - Q);
        do {
            old_convex_energy = convex_energy;

            // Get a Descent direction by minimising < \nabla E, s >
            descent_direction(desc, grad);

            // Solve for the best step size of the convex problem. It
            // is - frac{\phi^T(s-x) + 2 x^T \psi (s-x) + 2(x-x_old)^T
            // d (s-x)}{2 (s-x) (\psi+d) (s-x)}
            sx = desc - Q;

            psis.fill(0);
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
                pairwise_[k]->apply(tmp, sx);
                psis += tmp;
            }

            double num = dotProduct(sx, unary + 2*diag_dom.cwiseProduct(Q-Q_old), temp_dot) +
                2 * dotProduct(Q, psis, temp_dot);
            assert(num<=0); // This is negative if desc is really the good minimizer

            double denom = dotProduct(desc, psis + diag_dom.cwiseProduct(desc), temp_dot);
            assert(denom>0); // This is positive if we did our decomposition correctly

            double cst = dotProduct(Q, 0.5 * (grad + unary) - diag_dom.cwiseProduct(Q_old),temp_dot);

            optimal_step_size = - num/ (2 *denom);

            if (optimal_step_size > 1) {
                optimal_step_size = 1;
            } else if( optimal_step_size < 0){
                optimal_step_size = 0;
            }

            Q += optimal_step_size * sx;
            // Compute gradient of the convex problem at the new position
            grad += 2 * optimal_step_size * (psis-diag_dom.cwiseProduct(sx));

            // std::cout << "Coefficients: "<< denom << '\t' << num << '\t' << cst << '\n';
            convex_energy = pow(optimal_step_size, 2) * denom + optimal_step_size * num + cst;

            // energy = compute_energy(Q);
            convex_rounds++;
        } while ( (old_convex_energy - convex_energy) > 100 && convex_rounds<10 && optimal_step_size != 0);
        // We are now (almost) at a minimum of the convexified problem, so we
        // stop solving the convex problem and get a new convex approximation.


        //Check that the reduction in dotProduct actually corresponds to a decrease in runtime.


        // Compute our current value of the energy;
        // energy = compute_energy(Q);
        energy = dotProduct(Q, 0.5 * (grad + unary) - diag_dom.cwiseProduct(Q_old - Q), temp_dot);
        outer_rounds++;

        end = clock();
        perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        int_energy = assignment_energy_true(currentMap(Q));
        if (best_int_energy > int_energy) {
            best_int_energy = int_energy;
            best_Q = Q;
        }
        perf_energy = best_int_energy;
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;
        if (time_limit != 0 and total_time>time_limit) {
            break;
        }

    } while (((old_energy -energy) > 100 && outer_rounds < 20) or time_limit != 0);
    init = Q;
    return perfs;
}


void add_noise(MatrixXf & Q, float var) {
    Q += MatrixXf::Random(Q.rows(), Q.cols())*var;
    for(int col=0; col<Q.cols(); ++col) {
        Q.col(col) /= Q.col(col).sum();
    }
}

void print_distri(MatrixXf const & Q) {
    int nb_buckets = 20;
    VectorXi buckets(nb_buckets+1);
    buckets.fill(0);
    for(int label=0; label<Q.rows(); ++label) {
        for(int pixel=0; pixel<Q.cols();++pixel) {
            ++buckets[floor(Q(label, pixel)*nb_buckets)];
        }
        std::cout<<"Q distribution"<<std::endl;
        std::cout<<buckets.transpose()<<std::endl;
    }
}

void get_limited_indices(MatrixXf const & Q, std::vector<int> & indices) {
    VectorXd accum = Q.cast<double>().rowwise().sum();
    indices.clear();

    double represented = 0;
    int max_ind;
    while(represented < 0.99 * Q.cols() && indices.size() < Q.rows()) {
        int max_val = accum.maxCoeff(&max_ind);
        indices.push_back(max_ind);
        accum[max_ind] = -1e9;
        represented += max_val;
    }
}

MatrixXf get_restricted_matrix(MatrixXf const & in, std::vector<int> const & indices) {
    MatrixXf out(indices.size(), in.cols());
    out.fill(0);

    for(int i=0; i<indices.size(); i++) {
        out.row(i) = in.row(indices[i]);
    }

    return out;
}

MatrixXf get_extended_matrix(MatrixXf const & in, std::vector<int> const & indices, int max_rows) {
    MatrixXf out(max_rows, in.cols());
    out.fill(0);

    for(int i=0; i<indices.size(); i++) {
        out.row(indices[i]) = in.row(i);
    }

    return out;
}

void renormalize(MatrixXf & Q) {
    double sum;
    double uniform = 1.0/Q.rows();
    for(int i=0; i<Q.cols(); i++) {
        sum = Q.col(i).sum();
        if(sum == 0 || sum != sum) {
            Q.col(i).fill(uniform);
        } else {
            Q.col(i) /= sum;
        }
    }
}

// return the indices of pixels for a label cannot be determined with probability tol
void less_confident_pixels(std::vector<int> & indices, const MatrixXf & Q, float tol) {
    indices.clear();
    for (int i = 0; i < Q.cols(); ++i) {
        float max_prob = Q.col(i).maxCoeff();
        if (max_prob <= tol) indices.push_back(i);  // indices are in ascending order! (used later!!)
    }
}

// return the least confident (maximum probability of any label) k% of pixels
void less_confident_pixels2(std::vector<int> & indices, const MatrixXf & Q, float k = 10) {
    indices.clear();
    MatrixXf maxProb(1, Q.cols());
    MatrixXi ind(1, Q.cols());
    for (int i = 0; i < Q.cols(); ++i) {
        maxProb(1, i) = Q.col(i).maxCoeff();
    }
    sortRows(maxProb, ind);
    int rN = Q.cols() * k / 100;
    for (int i = Q.cols() - rN; i < Q.cols(); ++i) {
        indices.push_back(ind(1, i));    
    }
    std::sort(indices.begin(), indices.end());  // indices are in ascending order! (used later!!)
}

void update_restricted_matrix(MatrixXf & out, const MatrixXf & in, const std::vector<int> & pindices) {
    assert(out.cols() == pindices.size());
    out.fill(0);

    for(int i=0; i<pindices.size(); i++) {
        out.col(i) = in.col(pindices[i]);
    }
}

void update_extended_matrix(MatrixXf & out, const MatrixXf & in, const std::vector<int> & pindices) {
    assert(in.cols() == pindices.size());

    for(int i=0; i<pindices.size(); i++) {
        out.col(pindices[i]) = in.col(i);
    }
}

VectorXs get_original_label(VectorXs const & restricted_labels, std::vector<int> const & indices) {
    VectorXs extended_labels(restricted_labels.rows());
    for(int i=0; i<restricted_labels.rows(); i++) {
        extended_labels[i] = indices[restricted_labels[i]];
    }
    return extended_labels;
}

MatrixXf DenseCRF::lp_inference(MatrixXf & init, bool use_cond_grad, bool full_mat) const {
    // Random init to prevent too many elements to be 0
    // init.setRandom();
    // MatrixXf uns(init.rows(), init.cols());
    // uns.fill(1);
    // init = init + uns;
    // renormalize(init);

    // Restrict number of labels in the computation
    std::vector<int> indices;
    renormalize(init);
    if (full_mat) { // hack
        for(int i = 0; i < M_; ++i) indices.push_back(i);
    } else {
        get_limited_indices(init, indices);
    }
    int restricted_M = indices.size();
    MatrixXf unary = get_restricted_matrix(unary_->get(), indices);
    MatrixXf Q = get_restricted_matrix(init, indices);
    renormalize(Q);

    // std::cout<<"Using only "<<indices.size()<<" labels"<<std::endl;


    MatrixXf best_Q(restricted_M, N_), ones(restricted_M, N_), base_grad(restricted_M, N_), tmp(restricted_M, N_),
        grad(restricted_M, N_), tmp2(restricted_M, N_), desc(restricted_M, N_), tmp_single_line(1, N_);
    MatrixP dot_tmp(restricted_M, N_);
    MatrixXi ind(restricted_M, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    MatrixXf rescaled_Q(M_, N_);

    // Create copies of the original pairwise since we don't want normalization
    int nb_pairwise = pairwise_.size();
    PairwisePotential** no_norm_pairwise;
    no_norm_pairwise = (PairwisePotential**) malloc(pairwise_.size()*sizeof(PairwisePotential*));
    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        no_norm_pairwise[k] = new PairwisePotential(
            pairwise_[k]->features(),
            new PottsCompatibility(pairwise_[k]->parameters()(0)),
            pairwise_[k]->ktype(),
            NO_NORMALIZATION
            );
    }

    ones.fill(1);

    int i,j;

    best_Q = Q;
    double best_int_energy = assignment_energy(get_original_label(currentMap(Q), indices));
    printf("%3d: %f\n", 0, best_int_energy);

    // Compute the value of the energy
    double old_energy;
    assert(valid_probability(Q));
    double energy = 0;

    clock_t start, end;

    int it=0;
    do {
        ++it;
        old_energy = energy;

        // Compute the current energy and gradient
        // Unary
        energy = dotProduct(unary, Q, dot_tmp);
        grad = unary;

        // Pairwise
        sortRows(Q, ind);
        rescale(rescaled_Q, Q);
        for( unsigned int k=0; k<nb_pairwise; k++ ) {
            // Special case for 2 labels (used mainly for alpha expansion)
            if(false && Q.rows()==2) {
                // Compute only for one label
                no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp_single_line, ind.row(0));
                tmp2.fill(0);
                for(i=0; i<tmp.cols(); ++i) {
                    tmp2(0, ind(0, i)) = tmp_single_line(i);
                }
                // Closed form solution for the other label
                tmp2.row(1) = - tmp2.row(0);
            } else {
                // Add upper minus lower
                // The divide and conquer way
                start = clock();
                no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp, ind);
                tmp2.fill(0);
                for(i=0; i<tmp.cols(); ++i) {
                    for(j=0; j<tmp.rows(); ++j) {
                        tmp2(j, ind(j, i)) = tmp(j, i);
                    }
                }
                end = clock();
                float perf_timing = (double(end-start)/CLOCKS_PER_SEC);
                printf("DC: It: %d | id: %d | time: %f\n", it, k, perf_timing);

                // With the new discretized split computations
                // start = clock();
                // pairwise_[k]->apply_upper_minus_lower_ord(tmp2, rescaled_Q);
                /*end = clock();
                perf_timing = (double(end-start)/CLOCKS_PER_SEC);
                printf("ORD: It: %d | id: %d | time: %f\n", it, k, perf_timing);*/

                // The subgradients computed by new and old PH implementations are different 
                // (due to the equality handling)
                // Compare by computing the LP energy!!
                /*double tmp2_e = -dotProduct(tmp2, Q, dot_tmp);
                double tmp_e = -dotProduct(tmp, Q, dot_tmp);
                std::cout << "GT enegy: " << tmp2_e << ", new energy: " << tmp_e << " diff: " << abs(tmp2_e - tmp_e) << std::endl;*/
            }
                
            energy -= dotProduct(Q, tmp2, dot_tmp);
            grad -= tmp2;
        }
        // Print previous iteration energy
        // std::cout << it << ": " << energy << "\n";

        // Sub-gradient descent step
        double int_energy = 0;
        if(use_cond_grad) {
            descent_direction(desc, grad);

            float min = 0.0;
            float max = 1.0;
            double min_int_energy, max_int_energy, left_third_int_energy, right_third_int_energy;
            int split = 0;
            min_int_energy = assignment_energy(get_original_label(currentMap(Q), indices));
            max_int_energy = assignment_energy(get_original_label(currentMap(desc), indices));
            do {
                split++;
                double left_third = (2*min + max)/3.0;
                double right_third = (min + 2*max)/3.0;
                left_third_int_energy = assignment_energy(get_original_label(currentMap(Q+(desc-Q)*left_third), indices));
                right_third_int_energy = assignment_energy(get_original_label(currentMap(Q+(desc-Q)*right_third), indices));
                if(left_third_int_energy < right_third_int_energy) {
                    max = right_third;
                    max_int_energy = right_third_int_energy;
                    int_energy = left_third_int_energy;
                } else {
                    min = left_third;
                    min_int_energy = left_third_int_energy;
                    int_energy = right_third_int_energy;
                }
            } while(max-min > 0.001);
            //std::cout<<" learning rate: "<<(max+min)/2.0<<" expected: "<<min_int_energy<<std::endl;

            Q += 0.5*(max+min)*(desc - Q);
        } else {
            float min = 0.0;
            float max = 1e-3;
            double min_int_energy, max_int_energy, left_third_int_energy, right_third_int_energy;
            int split = 0;
            min_int_energy = assignment_energy(get_original_label(currentMap(Q), indices));
            max_int_energy = assignment_energy(get_original_label(currentMap(Q-max*grad), indices));
            do {
                split++;
                double left_third = (2*min + max)/3.0;
                double right_third = (min + 2*max)/3.0;
                left_third_int_energy = assignment_energy(get_original_label(currentMap(Q-left_third*grad), indices));
                right_third_int_energy = assignment_energy(get_original_label(currentMap(Q-right_third*grad), indices));
                if(left_third_int_energy < right_third_int_energy) {
                    max = right_third;
                    max_int_energy = right_third_int_energy;
                    int_energy = left_third_int_energy;
                } else {
                    min = left_third;
                    min_int_energy = left_third_int_energy;
                    int_energy = right_third_int_energy;
                }
            } while(max-min > 0.00001);

            Q -= 0.5*(max+min)*grad;

            // Project current estimates on valid space
            sortCols(Q, ind);
            for(int i=0; i<N_; ++i) {
                sum(i) = Q.col(i).sum()-1;
                K(i) = -1;
            }
            for(int i=0; i<N_; ++i) {
                for(int k=restricted_M; k>0; --k) {
                    double uk = Q(ind(k-1, i), i);
                    if(sum(i)/k < uk) {
                        K(i) = k;
                        break;
                    }
                    sum(i) -= uk;
                }
            }
            tmp.fill(0);
            for(int i=0; i<N_; ++i) {
                for(int k=0; k<restricted_M; ++k) {
                    tmp(k, i) = std::max(Q(k, i) - sum(i)/K(i), (double)0);
                }
            }
            Q = tmp;
        }

        if(int_energy < best_int_energy) {
            best_Q = Q;
            best_int_energy = int_energy;
        }
        renormalize(Q);
        printf("%3d: %f / %f / %f\n", it,energy, int_energy, best_int_energy);
        assert(valid_probability_debug(Q));
    } while(it<5);
    // std::cout <<"final projected energy: " << best_int_energy << "\n";
    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        delete no_norm_pairwise[k];
    }
    free(no_norm_pairwise);

    // Reconstruct an output with the correct number of labels
    best_Q = get_extended_matrix(best_Q, indices, M_);
    return best_Q;
}

// LP inference with simple gradient descent for new implementations
MatrixXf DenseCRF::lp_inference_new(MatrixXf & init) const {
    // Restrict number of labels in the computation
    MatrixXf Q = init;
    renormalize(Q);

    MatrixXf best_Q(M_, N_), ones(M_, N_), base_grad(M_, N_), tmp(M_, N_),
        grad(M_, N_), tmp2(M_, N_), desc(M_, N_), tmp_single_line(1, N_);
    MatrixP dot_tmp(M_, N_);
    MatrixXi ind(M_, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    MatrixXf unary = unary_->get();
    MatrixXf rescaled_Q(M_, N_);

    ones.fill(1);

    int i,j;
    int nb_pairwise = pairwise_.size();

    best_Q = Q;

    // Compute the value of the energy
    assert(valid_probability(Q));
    double energy = 0, best_energy = std::numeric_limits<double>::max(), best_int_energy = std::numeric_limits<double>::max();
    double int_energy = assignment_energy_true(currentMap(Q));
    energy = compute_energy_LP(Q);
    printf("Initial energy in the LP: %10.3f / %10.3f\n", energy, int_energy);

    bool adaptive = false;
	float delta_k = 1e6;
	float delta = 1e3;
	float beta = 0.5;
	float lambda = 1.5;
	float gamma_k = 1.9;
	float alpha_k = 1;
	double f_target = 0;

    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;

    int it=0;
    bool stop = false;
    do {
        ++it;

        // Compute the current energy and gradient
        // Unary
        energy = dotProduct(unary, Q, dot_tmp);
        grad = unary;

        // Pairwise
//        sortRows(Q, ind);
        start = std::chrono::high_resolution_clock::now();
        rescale(rescaled_Q, Q);
        for( unsigned int k=0; k<nb_pairwise; k++ ) {
            // Add upper minus lower
//            no_norm_pairwise_[k]->apply_upper_minus_lower_dc(tmp, ind);
//            tmp2.fill(0);
//            for(i=0; i<tmp.cols(); ++i) {
//                for(j=0; j<tmp.rows(); ++j) {
//                    tmp2(j, ind(j, i)) = tmp(j, i);
//                }
//            }
            pairwise_[k]->apply_upper_minus_lower_ord(tmp2, rescaled_Q);
            energy -= dotProduct(Q, tmp2, dot_tmp);
            grad -= tmp2;
        }
        end = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
        printf("# Time-PH: %f,\t", dt);
        
        if (adaptive) {
			if (it == 1) delta_k = energy/1.1;
			f_target = best_energy - delta_k;
			alpha_k = (float)(gamma_k * (energy - f_target)/dotProduct(grad, grad, dot_tmp));

			Q -= alpha_k * grad;	// subgrad-descent

			// update delta_k after projection					
			double new_energy = compute_energy_LP(Q);
			delta_k = (new_energy <= f_target) ? lambda * delta_k : std::max(beta * delta_k, delta);

            printf("%4d: %10.3f / %10.3f / %10.3f (%10.3f, %10.3f, %5.3f, %5.10f)\n", it-1, energy, 
					int_energy, best_energy, f_target, new_energy, delta_k, alpha_k);
		} else {
            // Print previous iteration energy
            //printf("%4d: %10.3f / %10.3f\n", it-1, energy, int_energy);
//          if( energy < best_energy) {
//                best_Q = Q;
//              best_energy = energy;
//          }

            // Sub-gradient descent step
            //Q -= grad/(it+1e5);

            // line-search
            start = std::chrono::high_resolution_clock::now();
            float min = 0.0;
            float max = 1e-1;
            double min_int_energy, max_int_energy, left_third_int_energy, right_third_int_energy;
            int split = 0;
            min_int_energy = assignment_energy_true(currentMap(Q));
            max_int_energy = assignment_energy_true(currentMap(Q-max*grad));
            do {
                split++;
                double left_third = (2*min + max)/3.0;
                double right_third = (min + 2*max)/3.0;
                left_third_int_energy = assignment_energy_true(currentMap(Q-left_third*grad));
                right_third_int_energy = assignment_energy_true(currentMap(Q-right_third*grad));
                if(left_third_int_energy < right_third_int_energy) {
                    max = right_third;
                    max_int_energy = right_third_int_energy;
                    int_energy = left_third_int_energy;
                } else {
                    min = left_third;
                    min_int_energy = left_third_int_energy;
                    int_energy = right_third_int_energy;
                }
            } while(max-min > 0.00001);
    		
            end = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
    
            printf("#Time-LS: %f, [%f, %f, %d]\t", dt, max,min, split);
            
			// sub-grad-step
			Q -= 0.5*(max+min)*grad;
    		//                
        }

        // Project current estimates on valid space
        sortCols(Q, ind);
        for(int i=0; i<N_; ++i) {
            sum(i) = Q.col(i).sum()-1;
            K(i) = -1;
        }
        for(int i=0; i<N_; ++i) {
            for(int k=M_; k>0; --k) {
                double uk = Q(ind(k-1, i), i);
                if(sum(i)/k < uk) {
                    K(i) = k;
                    break;
                }
                sum(i) -= uk;
            }
        }
        tmp.fill(0);
        for(int i=0; i<N_; ++i) {
            for(int k=0; k<M_; ++k) {
                tmp(k, i) = std::min(std::max(Q(k, i) - sum(i)/K(i), 0.0), 1.0);
            }
        }
        Q = tmp;

        int_energy = assignment_energy_true(currentMap(Q));
        stop = (abs(best_int_energy-int_energy) < 1000);
        if(int_energy < best_int_energy) {
            best_Q = Q;
            best_int_energy = int_energy;
        }
        printf("#IT: %f / %f / %f\n", energy, int_energy, best_int_energy);

        renormalize(Q);
        assert(valid_probability_debug(Q));
    } while(it<5 && !stop);
    std::cout <<"final projected energy: " << int_energy << "\n";

    return best_Q;
}

// Project current estimates on valid space
void feasible_Q(MatrixXf & tmp, MatrixXi & ind, VectorXd & sum, VectorXi & K, const MatrixXf & Q) {
    sortCols(Q, ind);
    for(int i=0; i<Q.cols(); ++i) {
        sum(i) = Q.col(i).sum()-1;
        K(i) = -1;
    }
    for(int i=0; i<Q.cols(); ++i) {
        for(int k=Q.rows(); k>0; --k) {
            double uk = Q(ind(k-1, i), i);
            if(sum(i)/k < uk) {
                K(i) = k;
                break;
            }
            sum(i) -= uk;
        }
    }
    tmp.fill(0);
    for(int i=0; i<Q.cols(); ++i) {
        for(int k=0; k<Q.rows(); ++k) {
            tmp(k, i) = std::min(std::max(Q(k, i) - sum(i)/K(i), 0.0), 1.0);
        }
    }
}

// make a step of qp_gamma: -- \cite{NNQP solver Xiao and Chen 2014} - O(n) implementation!!
void qp_gamma_step(VectorXf & v_gamma, const VectorXf & v_pos_h, const VectorXf & v_neg_h, const float qp_delta, 
        const int M, const float lambda, VectorXf & v_step, VectorXf & v_tmp, VectorXf & v_tmp2) {
    // C: lambda*I - (lambda/m)*ones
    // neg_C: max(-C, 0)- elementwise
    // abs_C: abs(C)- elementwise
    float sum = v_gamma.sum();
    v_tmp2.fill(sum);
    
    v_tmp = v_tmp2 - v_gamma;
    v_tmp *= (2*lambda/float(M));  // 2 * neg_C * v_gamma
    v_step = v_tmp + v_pos_h;
	v_step = v_step.array() + qp_delta;

    v_tmp = (1.0/float(M)) * v_tmp2 + (1-2.0/float(M)) * v_gamma;
    v_tmp *= lambda;    // abs_C * v_gamma
	v_tmp = v_tmp + v_neg_h;
	v_tmp = v_tmp.array() + qp_delta;
	v_step = v_step.cwiseQuotient(v_tmp);
	v_gamma = v_gamma.cwiseProduct(v_step);
}

// multiply by C in linear time!
void qp_gamma_multiplyC(VectorXf & v_out, const VectorXf & v_in, const int M, const float lambda) {
    // C: lambda*I - (lambda/m)*ones
    float sum = v_in.sum();
    v_out.fill(sum);
    v_out = (-1.0/float(M)) * v_out + v_in;
    v_out *= lambda;
}

// LP inference with proximal algorithm with restricted pixel and labels
MatrixXf DenseCRF::lp_inference_prox_restricted(MatrixXf & init, LP_inf_params & params) const {

    MatrixXf best_Q(M_, N_), tmp(M_, N_), tmp2(M_, N_);
    MatrixP dot_tmp(M_, N_);
    MatrixXi ind(M_, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    MatrixXf unary = unary_->get();

    MatrixXf Q = init;
    renormalize(Q);
    assert(valid_probability_debug(Q));
    best_Q = Q;

    // Compute the value of the energy
    double energy = 0, best_energy = std::numeric_limits<double>::max(), 
		   best_int_energy = std::numeric_limits<double>::max();
    double int_energy = assignment_energy_true(currentMap(Q));
	best_int_energy = int_energy;
#if VERBOSE    
    double kl = klDivergence(Q, max_rounding(Q));
	energy = compute_energy_LP(Q);
	best_energy = energy;
    printf("Initial energy in the LP: %10.3f / %10.3f / %10.3f\n", energy, int_energy, kl);
#endif

	const int maxiter = params.prox_max_iter;
    const bool best_int = params.best_int;
    const bool accel_prox = params.accel_prox;
	const float prox_tol = params.prox_energy_tol;		// proximal energy tolerance
    const float confidence_tol = params.confidence_tol;

	// dual Frank-Wolfe variables
	const float dual_gap_tol = params.dual_gap_tol;		// dual gap tolerance
	const float lambda = params.prox_reg_const;	// proximal-regularization constant
	const int fw_maxiter = params.fw_max_iter;
	float delta = 1;						// FW step size
	double dual_gap = 0, dual_energy = 0;

    // gamma-qp variables
    const float qp_delta = params.qp_const;	// constant used in qp-gamma
    const float qp_tol = params.qp_tol;		// qp-gamma tolernace
    const int qp_maxiter = params.qp_max_iter;

    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    
    std::vector<int> pI;    // restricted pixels

    MatrixXf cur_Q(M_, N_);		// current Q in prox step
#if VERBOSE
  	MatrixXf int_Q(M_, N_);		// store integral Q
#endif
    // accelerated prox_lp
    MatrixXf prev_Q(M_, N_);	// Q resulted in prev prox step
    prev_Q.fill(0);
    float w_it = 1;             // momentum weight: eg. it/(it+3)

    int it=0;
    int count = 0;
    do {
        ++it;

        // matrix creations
        less_confident_pixels(pI, Q, confidence_tol);
        //float percent = 10;
        //less_confident_pixels2(pI, Q, percent);
        int rN = pI.size();
        double percent = double(rN)/double(Q.cols())*100;
        if (percent < 1.0) {
            std::cout << "#CONV: Less confident pixels are less than 1%, exiting...\n";
            break;
        }
#if VERBOSE    
        std::cout << "No of pixels with probability less than " << confidence_tol << " is: " << pI.size() 
            << " out of " << Q.cols() << ", percentage: " << double(pI.size())/double(Q.cols())*100 << "%" << std::endl;
#endif
        // dual variables
        MatrixXf alpha_tQ(M_, rN);	// A * alpha, (t - tilde not iteration)
        MatrixXf s_tQ(M_, rN);		// A * s, conditional gradient of FW == subgradient
        VectorXf beta(rN);			// unconstrained --> correct beta values (beta.row(i) == v_beta forall i)
        MatrixXf beta_mat(M_, rN);	// beta_mat.row(i) == beta forall i --> N_ * M_ elements 
        MatrixXf gamma(M_, rN);		// nonnegative
        
        MatrixXf cur_rQ(M_, rN);		// current Q in prox step
        MatrixXf rescaled_rQ(M_, rN);   // infeasible Q rescaled to be within [0,1]
        MatrixXf rtmp(M_, rN), rtmp2(M_, rN);
        MatrixP rdot_tmp(M_, rN);
    
        MatrixXf rQ(M_, rN);		    // restricted Q in prox step
        update_restricted_matrix(rQ, Q, pI);
        MatrixXf runary(M_, rN);        // restricted unary
        update_restricted_matrix(runary, unary, pI);
        
	    //MatrixXf C(M_, M_), neg_C(M_, M_), pos_C(M_, M_), abs_C(M_, M_);
        VectorXf v_gamma(M_), v_y(M_), v_pos_h(M_), v_neg_h(M_), v_step(M_), v_tmp(M_), v_tmp2(M_);
        MatrixXf Y(M_, rN), neg_H(M_, rN), pos_H(M_, rN);
        VectorXf qp_values(rN);

//        pos_C = MatrixXf::Identity(M_, M_) * (1-1.0/float(M_));				      
//        pos_C *= lambda;	// pos_C
//        neg_C = MatrixXf::Ones(M_, M_) - MatrixXf::Identity(M_, M_);	
//        neg_C /= float(M_);													      
//        neg_C *= lambda; 	// neg_C
//        C = pos_C - neg_C;	// C
//        abs_C = pos_C + neg_C;	// abs_C

		// initialization
		cur_Q = Q;
        if (accel_prox) {   // accelerated prox
            w_it = float(it)/(it+3.0);  // simplest choice
            tmp = Q - prev_Q;
            tmp *= w_it;
            cur_Q += tmp;   // cur_Q = Q + w_it(Q - prev_Q)
            prev_Q = Q;
        }
        
		int pit = 0;
		alpha_tQ.fill(0);	// all zero alpha_tQ is feasible --> alpha^1_{abi} = alpha^2_{abi} = K_{ab}/4
		beta_mat.fill(0);
		beta.fill(0);
		gamma.fill(1);	// all zero is a fixed point of QP iteration!
        update_restricted_matrix(cur_rQ, cur_Q, pI);
		
		// prox step
		do {	
			++pit;
			// initialization
			s_tQ.fill(0);

            //if (pit == 1) { // only compute beta and gamma in the first iteration
			// QP-gamma -- \cite{NNQP solver Xiao and Chen 2014}
			// case-1: solve for gamma using qp solver! 
			// 1/2 * gamma^T * C * gamma - gamma^T * H
			// populate Y matrix
			rtmp = alpha_tQ - runary;
			for (int i = 0; i < rN; ++i) {
//                v_y = C * rtmp.col(i);
                // do it in linear time
                v_tmp2 = rtmp.col(i);
                qp_gamma_multiplyC(v_y, v_tmp2, M_, lambda);
				Y.col(i) = v_y;
			}	
			Y += cur_rQ;		// H = -Y
			for (int i = 0; i < rN; ++i) {
				for (int j = 0; j < M_; ++j) {	
					pos_H(j, i) = std::max(-Y(j, i), (float)0);		// pos_H 
					neg_H(j, i) = std::max(Y(j, i), (float)0);		// neg_H
				}
			}
			// qp iterations, 
			int qpit = 0;
			qp_values.fill(0);
		    //gamma.fill(1);	// initializing gamma here affects efficiency!
			float qp_value = std::numeric_limits<float>::max();
#if VERBOSE
            //start = clock();
            start = std::chrono::high_resolution_clock::now();
#endif
			do {
				//solve for each pixel separately
				for (int i = 0; i < rN; ++i) {
					v_gamma = gamma.col(i);
					v_pos_h = pos_H.col(i);
					v_neg_h = neg_H.col(i);
                    //
//                    v_step = 2 * neg_C * v_gamma + v_pos_h;
//					v_step = v_step.array() + qp_delta;
//					v_tmp = abs_C * v_gamma + v_neg_h;
//				    v_tmp = v_tmp.array() + qp_delta;
//					v_step = v_step.cwiseQuotient(v_tmp);
//					v_gamma = v_gamma.cwiseProduct(v_step);
                    // do it in linear time
                    qp_gamma_step(v_gamma, v_pos_h, v_neg_h, qp_delta, M_, lambda, v_step, v_tmp, v_tmp2);
					gamma.col(i) = v_gamma;

					// qp value
//                    v_tmp = C * v_gamma;
                    // do it in linear time
                    qp_gamma_multiplyC(v_tmp, v_gamma, M_, lambda);
					v_y = Y.col(i);
					qp_values(i) = 0.5 * v_gamma.dot(v_tmp) + v_gamma.dot(v_y);
				}
				float qp_value1 = qp_values.sum();
				//printf("\n#QP: %4d, %10.3f", qpit, qp_value1);
				++qpit;
				if (std::abs(qp_value - qp_value1) < qp_tol) break;
				qp_value = qp_value1;
			} while (qpit < qp_maxiter);
#if VERBOSE
            //end = clock();
            //double dt = (double)(end-start)/CLOCKS_PER_SEC
            end = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
            printf("# Time-QP %d: %5.5f, %10.3f\t", qpit, dt, qp_value);
#endif
			// end-qp-gamma

			// case-2: update beta -- gradient of dual wrt beta equals to zero
			beta_mat = (alpha_tQ + gamma - runary);	// -B^T/l * (A * alpha + gamma - phi)
			// DON'T DO IT AT ONCE!! (RETURNS A SCALAR???)--> do it in two steps
			beta = -beta_mat.colwise().sum();	
			beta /= M_;
			// repeat beta in each row of beta_mat - - B * beta
			for (int j = 0; j < M_; ++j) {
				beta_mat.row(j) = beta;
			}
            //}

			// case-3: dual conditional-gradient or primal-subgradient (do it as the final case)
			rQ = lambda * (alpha_tQ + beta_mat + gamma - runary) + cur_rQ;	// Q may be infeasible --> but no problem

			// new PH implementation doesn't work with Q values outside [0,1] - or in fact truncates to be within [0,1]
    		// rescale Q to be within [0,1] -- order of Q values preserved!
    		rescale(rescaled_rQ, rQ);

			// subgradient lower minus upper
			// Pairwise
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
#if VERBOSE
                //start = clock();
                start = std::chrono::high_resolution_clock::now();
#endif
				// new PH implementation
				// rescaled_Q values in the range [0,1] --> but the same order as Q! --> subgradient of Q
                bool store = (pit == 1); // store only at the first iteration
                pairwise_[k]->apply_upper_minus_lower_ord_restricted(rtmp, rescaled_rQ, pI, Q, store);	

                s_tQ += rtmp;	// A * s is lower minus upper, keep neg introduced by compatibility->apply
#if VERBOSE
                //end = clock();
                //double dt = (double)(end-start)/CLOCKS_PER_SEC;
                end = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
                printf("# Time-%d: %5.5f\t", k, dt);
#endif
            }
			// find dual gap
			rtmp = alpha_tQ - s_tQ;	
			dual_gap = dotProduct(rtmp, rQ, rdot_tmp);

#if VERBOSE
			// dual-energy value
			rtmp2 = rQ - cur_rQ;
			dual_energy = dotProduct(rtmp2, rtmp2, rdot_tmp) / (2* lambda) + dotProduct(rtmp2, cur_rQ, rdot_tmp) / lambda;
			dual_energy -= beta.sum();
			double primal_energy = dotProduct(rtmp2, rtmp2, rdot_tmp) / (2* lambda);
		   	primal_energy += dotProduct(runary, rQ, rdot_tmp);
			primal_energy -= dotProduct(s_tQ, rQ, rdot_tmp);	// cancel the neg in s_tQ
			//assert(dual_gap == (dual_energy - primal_energy));
			printf("%4d: [%10.3f = %10.3f, %10.3f, %10.3f, ", pit-1, dual_gap, primal_energy+dual_energy, 
                    -dual_energy, primal_energy);
#endif
			if (dual_gap <= dual_gap_tol) break;	// stopping condition

			// optimal fw step size
			delta = (float)(dual_gap / (lambda * dotProduct(rtmp, rtmp, rdot_tmp)));
			delta = std::min(std::max(delta, (float)0.0), (float)1.0);  // I may not need to truncate the step-size!!
			assert(delta >= 0);
#if VERBOSE
			printf("%1.10f]\n", delta);
#endif
			// update alpha_tQ
			rtmp = s_tQ - alpha_tQ;
			rtmp *= delta;
			alpha_tQ += rtmp;	// alpha_tQ = alpha_tQ + delta * (s_tQ - alpha_tQ);

		} while(pit<fw_maxiter);

        update_extended_matrix(Q, rQ, pI);
		// project Q back to feasible region
		feasible_Q(tmp, ind, sum, K, Q);
		Q = tmp;
		renormalize(Q);
        assert(valid_probability_debug(Q));

        double prev_int_energy = int_energy;
        int_energy = assignment_energy_true(currentMap(Q));
#if VERBOSE
        double prev_energy = energy;
		energy = compute_energy_LP(Q);
//#if VERBOSE
        int_Q = max_rounding(Q);
        kl = klDivergence(Q, int_Q);
//#endif
#endif
        if (best_int) {
            if ((best_int_energy - int_energy) < prox_tol) ++count; // may also increase!
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: best_int_energy - int_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
		    if(int_energy < best_int_energy) {
                best_Q = Q;
                best_int_energy = int_energy;
            }
#if VERBOSE
            printf("%4d: %10.3f / %10.3f / %10.3f / %10.3f\n", it-1, energy, int_energy, kl, best_int_energy);
#endif
        } else {
#if VERBOSE == false
            double prev_energy = energy;
    		energy = compute_energy_LP(Q);
#endif
            if (abs(energy - prev_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: energy - prev_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
    		if( energy < best_energy) {
                best_Q = Q;
                best_energy = energy;
            }
#if VERBOSE
            printf("%4d: %10.3f / %10.3f / %10.3f / %10.3f\n", it-1, energy, int_energy, kl, best_energy);
#endif
        }

    } while(it<maxiter);

    return best_Q;
}

// LP inference with proximal algorithm
MatrixXf DenseCRF::lp_inference_prox(MatrixXf & init, LP_inf_params & params) const {

    MatrixXf best_Q(M_, N_), tmp(M_, N_), tmp2(M_, N_);
    MatrixP dot_tmp(M_, N_);
    MatrixXi ind(M_, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    MatrixXf unary = unary_->get();

    MatrixXf Q = init;
    renormalize(Q);
    assert(valid_probability_debug(Q));
    best_Q = Q;

    // Compute the value of the energy
    double energy = 0, best_energy = std::numeric_limits<double>::max(), 
		   best_int_energy = std::numeric_limits<double>::max();
    double int_energy = assignment_energy_true(currentMap(Q));
#if VERBOSE    
    double kl = klDivergence(Q, max_rounding(Q));
#endif
	energy = compute_energy_LP(Q);
    if (energy > int_energy) {  // choose the best initialization 
        Q = max_rounding(Q);
    	energy = compute_energy_LP(Q);
    }
	best_energy = energy;
	best_int_energy = int_energy;
#if VERBOSE    
    printf("Initial energy in the LP: %10.3f / %10.3f / %10.3f\n", energy, int_energy, kl);
#endif

	const int maxiter = params.prox_max_iter;
    const bool best_int = params.best_int;
    const bool accel_prox = params.accel_prox;
	const float prox_tol = params.prox_energy_tol;		// proximal energy tolerance

	// dual Frank-Wolfe variables
	const float dual_gap_tol = params.dual_gap_tol;		// dual gap tolerance
	const float lambda = params.prox_reg_const;	// proximal-regularization constant
	const int fw_maxiter = params.fw_max_iter;
	float delta = 1;						// FW step size
	double dual_gap = 0, dual_energy = 0;

	// dual variables
	MatrixXf alpha_tQ(M_, N_);	// A * alpha, (t - tilde not iteration)
	MatrixXf s_tQ(M_, N_);		// A * s, conditional gradient of FW == subgradient
	VectorXf beta(N_);			// unconstrained --> correct beta values (beta.row(i) == v_beta forall i)
	MatrixXf beta_mat(M_, N_);	// beta_mat.row(i) == beta forall i --> N_ * M_ elements 
	MatrixXf gamma(M_, N_);		// nonnegative
	
	MatrixXf cur_Q(M_, N_);		// current Q in prox step
	MatrixXf rescaled_Q(M_, N_);// infeasible Q rescaled to be within [0,1]
#if VERBOSE
	MatrixXf int_Q(M_, N_);		// store integral Q
#endif

    // accelerated prox_lp
	MatrixXf prev_Q(M_, N_);	// Q resulted in prev prox step
    prev_Q.fill(0);
    float w_it = 1;             // momentum weight: eg. it/(it+3)

	// gamma-qp variables
	const float qp_delta = params.qp_const;	// constant used in qp-gamma
	const float qp_tol = params.qp_tol;		// qp-gamma tolernace
	const int qp_maxiter = params.qp_max_iter;

	//MatrixXf C(M_, M_), neg_C(M_, M_), pos_C(M_, M_), abs_C(M_, M_);
	VectorXf v_gamma(M_), v_y(M_), v_pos_h(M_), v_neg_h(M_), v_step(M_), v_tmp(M_), v_tmp2(M_);
	MatrixXf Y(M_, N_), neg_H(M_, N_), pos_H(M_, N_);
	VectorXf qp_values(N_);

//	pos_C = MatrixXf::Identity(M_, M_) * (1-1.0/float(M_));				      
//	pos_C *= lambda;	// pos_C
//	neg_C = MatrixXf::Ones(M_, M_) - MatrixXf::Identity(M_, M_);	
//	neg_C /= float(M_);													      
//	neg_C *= lambda; 	// neg_C
//	C = pos_C - neg_C;	// C
//	abs_C = pos_C + neg_C;	// abs_C

    //clock_t start, end;
    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    
//    // multi-plane FW --> NO ADAPTIVE SELECTION OF WORKING-SET-SIZE AND APPROX-FW-ITER as of now
//    const int work_set_size = params.work_set_size;
//    const int approx_fw_iter = params.approx_fw_iter;
//    const bool mp_fw = (work_set_size != 0 || approx_fw_iter != 0);
//    std::vector<MatrixXf> working_set;    // stores conditional gradients (copied)
//    double dual_prox_start = 0, dual_app_start = 0;
//    //clock_t time_prox_start;
//    htime time_prox_start;

    int it=0;
    int count = 0;
    do {
        ++it;

		// initialization
		beta_mat.fill(0);
		beta.fill(0);
		gamma.fill(1);	// all zero is a fixed point of QP iteration!
		cur_Q = Q;
        
        if (accel_prox) {   // accelerated prox
            w_it = float(it)/(it+3.0);  // simplest choice
            tmp = Q - prev_Q;
            tmp *= w_it;
            cur_Q += tmp;   // cur_Q = Q + w_it(Q - prev_Q)
            prev_Q = Q;
        }
        
		int pit = 0;
		alpha_tQ.fill(0);	// all zero alpha_tQ is feasible --> alpha^1_{abi} = alpha^2_{abi} = K_{ab}/4
		
		// prox step
		do {	
			++pit;
//            if (mp_fw) {
//                //time_prox_start = clock();
//                time_prox_start = std::chrono::high_resolution_clock::now();
//            }
			// initialization
			s_tQ.fill(0);

            //if (pit == 1) { // only compute beta and gamma in the first iteration
			// QP-gamma -- \cite{NNQP solver Xiao and Chen 2014}
			// case-1: solve for gamma using qp solver! 
			// 1/2 * gamma^T * C * gamma - gamma^T * H
			// populate Y matrix
			tmp = alpha_tQ - unary;
			for (int i = 0; i < N_; ++i) {
//                v_y = C * tmp.col(i);
                // do it in linear time
                v_tmp2 = tmp.col(i);
                qp_gamma_multiplyC(v_y, v_tmp2, M_, lambda);
				Y.col(i) = v_y;
			}	
			Y += cur_Q;		// H = -Y
			for (int i = 0; i < N_; ++i) {
				for (int j = 0; j < M_; ++j) {	
					pos_H(j, i) = std::max(-Y(j, i), (float)0);		// pos_H 
					neg_H(j, i) = std::max(Y(j, i), (float)0);		// neg_H
				}
			}
			// qp iterations, 
			int qpit = 0;
			qp_values.fill(0);
		    //gamma.fill(1);	// initializing gamma here affects efficiency!
			float qp_value = std::numeric_limits<float>::max();
#if VERBOSE
            //start = clock();
            start = std::chrono::high_resolution_clock::now();
#endif
			do {
				//solve for each pixel separately
				for (int i = 0; i < N_; ++i) {
					v_gamma = gamma.col(i);
					v_pos_h = pos_H.col(i);
					v_neg_h = neg_H.col(i);
                    //
//                    v_step = 2 * neg_C * v_gamma + v_pos_h;
//					v_step = v_step.array() + qp_delta;
//					v_tmp = abs_C * v_gamma + v_neg_h;
//				    v_tmp = v_tmp.array() + qp_delta;
//					v_step = v_step.cwiseQuotient(v_tmp);
//					v_gamma = v_gamma.cwiseProduct(v_step);
                    // do it linear time
                    qp_gamma_step(v_gamma, v_pos_h, v_neg_h, qp_delta, M_, lambda, v_step, v_tmp, v_tmp2);
					gamma.col(i) = v_gamma;

					// qp value
//                    v_tmp = C * v_gamma;
                    // do it in linear time
                    qp_gamma_multiplyC(v_tmp, v_gamma, M_, lambda);
					v_y = Y.col(i);
					qp_values(i) = 0.5 * v_gamma.dot(v_tmp) + v_gamma.dot(v_y);
				}
				float qp_value1 = qp_values.sum();
				//printf("\n#QP: %4d, %10.3f", qpit, qp_value1);
				++qpit;
				if (std::abs(qp_value - qp_value1) < qp_tol) break;
				qp_value = qp_value1;
			} while (qpit < qp_maxiter);
#if VERBOSE
            //end = clock();
            //double dt = (double)(end-start)/CLOCKS_PER_SEC
            end = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
            printf("# Time-QP %d: %5.5f, %10.3f\t", qpit, dt, qp_value);
#endif
			// end-qp-gamma

			// case-2: update beta -- gradient of dual wrt beta equals to zero
			beta_mat = (alpha_tQ + gamma - unary);	// -B^T/l * (A * alpha + gamma - phi)
			// DON'T DO IT AT ONCE!! (RETURNS A SCALAR???)--> do it in two steps
			beta = -beta_mat.colwise().sum();	
			beta /= M_;
			// repeat beta in each row of beta_mat - - B * beta
			for (int j = 0; j < M_; ++j) {
				beta_mat.row(j) = beta;
			}
            //}

			// case-3: dual conditional-gradient or primal-subgradient (do it as the final case)
			Q = lambda * (alpha_tQ + beta_mat + gamma - unary) + cur_Q;	// Q may be infeasible --> but no problem
#if BRUTE_FORCE
        	sortRows(Q, ind);
#else
			// new PH implementation doesn't work with Q values outside [0,1] - or in fact truncates to be within [0,1]
    		// rescale Q to be within [0,1] -- order of Q values preserved!
    		rescale(rescaled_Q, Q);
//            // check colinearity of subgradients
//            double ph_e = 0, bf_e = 0;
//            compare_energies(rescaled_Q, ph_e, bf_e, false, false, true);
//            //
#endif
			// subgradient lower minus upper
			// Pairwise
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
#if VERBOSE
                //start = clock();
                start = std::chrono::high_resolution_clock::now();
#endif
#if BRUTE_FORCE
				// brute-force computation
            	no_norm_pairwise_[k]->apply_upper_minus_lower_bf(tmp2, ind);
            	//no_norm_pairwise_[k]->apply_upper_minus_lower_dc(tmp2, ind);
            	for(int i=0; i<tmp2.cols(); ++i) {
                	for(int j=0; j<tmp2.rows(); ++j) {
                    	tmp(j, ind(j, i)) = tmp2(j, i);
                	}
            	}
#else
				// new PH implementation
				// rescaled_Q values in the range [0,1] --> but the same order as Q! --> subgradient of Q
                pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_Q);	
#endif
                s_tQ += tmp;	// A * s is lower minus upper, keep neg introduced by compatibility->apply
#if VERBOSE
                //end = clock();
                //double dt = (double)(end-start)/CLOCKS_PER_SEC;
                end = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
                printf("# Time-%d: %5.5f\t", k, dt);
#endif
            }
			// find dual gap
			tmp = alpha_tQ - s_tQ;	
			dual_gap = dotProduct(tmp, Q, dot_tmp);

#if VERBOSE
			// dual-energy value
			tmp2 = Q - cur_Q;
			dual_energy = dotProduct(tmp2, tmp2, dot_tmp) / (2* lambda) + dotProduct(tmp2, cur_Q, dot_tmp) / lambda;
			dual_energy -= beta.sum();
			double primal_energy = dotProduct(tmp2, tmp2, dot_tmp) / (2* lambda);
		   	primal_energy += dotProduct(unary, Q, dot_tmp);
			primal_energy -= dotProduct(s_tQ, Q, dot_tmp);	// cancel the neg in s_tQ
			//assert(dual_gap == (dual_energy - primal_energy));
			printf("%4d: [%10.3f = %10.3f, %10.3f, %10.3f, ", pit-1, dual_gap, primal_energy+dual_energy, 
                    -dual_energy, primal_energy);
//            if (mp_fw) dual_prox_start = -dual_energy;
            //if (dual_gap < 0) {   // may become negative due to PH approximations!
            //    std::cout << "\nERROR: Dual-gap cannot be negative!\n";
            //    exit(1);
            //}
#endif
			if (dual_gap <= dual_gap_tol) break;	// stopping condition

			// optimal fw step size
			delta = (float)(dual_gap / (lambda * dotProduct(tmp, tmp, dot_tmp)));
			delta = std::min(std::max(delta, (float)0.0), (float)1.0);  // I may not need to truncate the step-size!!
			assert(delta > 0);
#if VERBOSE
			printf("%1.10f]\n", delta);
#endif
			// update alpha_tQ
			tmp = s_tQ - alpha_tQ;
			tmp *= delta;
			alpha_tQ += tmp;	// alpha_tQ = alpha_tQ + delta * (s_tQ - alpha_tQ);

//            if (mp_fw) {
//                if (working_set.size() == work_set_size) working_set.erase(working_set.begin());
//                working_set.push_back(s_tQ);
//            }
//
//            // multi-plane approximate iterations
//            for (int appit = 0; appit < approx_fw_iter; ++appit) {
//                // case-1: use the old-gamma
//                // case-2:
//                beta_mat = (alpha_tQ + gamma - unary);	// -B^T/l * (A * alpha + gamma - phi)
//			    beta = -beta_mat.colwise().sum();	
//			    beta /= M_;
//			    // repeat beta in each row of beta_mat - - B * beta
//			    for (int j = 0; j < M_; ++j) {
//				    beta_mat.row(j) = beta;
//			    }
//                //
//                // case-3
//                Q = lambda * (alpha_tQ + beta_mat + gamma - unary) + cur_Q;
//
//                MatrixXf& s_tQ_hat = working_set[0];
//                double maxe = dotProduct(s_tQ_hat, Q, dot_tmp);
//#if VERBOSE
//                //clock_t st = clock();
//                htime st = std::chrono::high_resolution_clock::now();
//#endif
//                for (int i = 1; i < working_set.size(); ++i) {
//                    double e = dotProduct(working_set[i], Q, dot_tmp);
//                    if (maxe < e) {
//                        maxe = e;
//                        s_tQ_hat = working_set[i];
//                    }
//                    //std::cout << "##i: " << i << ", e: " << e << ", maxe: " << maxe << std::endl;
//                }
//#if VERBOSE
//                //clock_t et = clock();
//                //double app_fw_time = (double)(et-st)/CLOCKS_PER_SEC;
//                htime et = std::chrono::high_resolution_clock::now();
//                double app_fw_time = std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
//                printf("#App-FW-Time: %5.5f, size: %d\t", app_fw_time, (int)working_set.size());
//#endif
//
//                // find dual gap
//    			tmp = alpha_tQ - s_tQ_hat;	
//    			dual_gap = dotProduct(tmp, Q, dot_tmp);
//    
//#if VERBOSE
//    			// dual-energy value
//    			tmp2 = Q - cur_Q;
//    			dual_energy = dotProduct(tmp2, tmp2, dot_tmp) / (2* lambda) + dotProduct(tmp2, cur_Q, dot_tmp) / lambda;
//    			dual_energy -= beta.sum();
//    			double primal_energy = dotProduct(tmp2, tmp2, dot_tmp) / (2* lambda);
//    		   	primal_energy += dotProduct(unary, Q, dot_tmp);
//    			primal_energy -= dotProduct(s_tQ_hat, Q, dot_tmp);	// cancel the neg in s_tQ_hat
//    			//assert(dual_gap == (dual_energy - primal_energy));
//    			printf("%4d: [%10.3f = %10.3f, %10.3f, %10.3f, ", appit, dual_gap, primal_energy+dual_energy, 
//                        -dual_energy, primal_energy);
//                if (appit > 0) {
//                    //double prox_tot_time = (double)(et-time_prox_start)/CLOCKS_PER_SEC;
//                    double prox_tot_time = std::chrono::duration_cast<std::chrono::duration<double>>
//                        (et-time_prox_start).count();
//                    double avg_imp = (-dual_energy-dual_prox_start)/prox_tot_time;
//                    double app_imp = (-dual_energy-dual_app_start)/app_fw_time;
//                    printf("(%10.3f, %10.3f), ", avg_imp, app_imp);
//                    if (avg_imp >= app_imp) {
//                        printf(" break: app_imp\n");
//                        break;
//                    }
//                }
//                dual_app_start = -dual_energy;
//#endif
//    			if (dual_gap <= 10) {
//                    printf(" break: dual_gap\n");
//                    break;	// stopping condition
//                }
//    
//    			// optimal fw step size
//    			delta = (float)(dual_gap / (lambda * dotProduct(tmp, tmp, dot_tmp)));
//    			delta = std::min(std::max(delta, (float)0.0), (float)1.0);
//    			assert(delta > 0);
//#if VERBOSE
//    			printf("%1.10f]\n", delta);
//#endif
//
//                // update alpha_tQ
//		    	tmp = s_tQ_hat - alpha_tQ;
//	    		tmp *= delta;
//    			alpha_tQ += tmp;	// alpha_tQ = alpha_tQ + delta * (s_tQ_hat - alpha_tQ);
//            }

		} while(pit<fw_maxiter);

		// project Q back to feasible region
		feasible_Q(tmp, ind, sum, K, Q);
		Q = tmp;
		renormalize(Q);
        assert(valid_probability_debug(Q));

        double prev_int_energy = int_energy;
        int_energy = assignment_energy_true(currentMap(Q));
#if VERBOSE
        double prev_energy = energy;
		energy = compute_energy_LP(Q);
//#if VERBOSE
        int_Q = max_rounding(Q);
        kl = klDivergence(Q, int_Q);
//#endif
#endif
        if (best_int) {
            if (abs(int_energy - prev_int_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: int_energy - prev_int_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
		    if(int_energy < best_int_energy) {
                best_Q = Q;
                best_int_energy = int_energy;
            }
#if VERBOSE
            printf("%4d: %10.3f / %10.3f / %10.3f / %10.3f\n", it-1, energy, int_energy, kl, best_int_energy);
#endif
        } else {
#if VERBOSE == false
            double prev_energy = energy;
    		energy = compute_energy_LP(Q);
#endif
            if (abs(energy - prev_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: energy - prev_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
    		if( energy < best_energy) {
                best_Q = Q;
                best_energy = energy;
            }
#if VERBOSE
            printf("%4d: %10.3f / %10.3f / %10.3f / %10.3f\n", it-1, energy, int_energy, kl, best_energy);
#endif
        }
        if (params.less_confident_percent > 0) {
            float confidence_tol = params.confidence_tol;
            std::vector<int> pI;
            less_confident_pixels(pI, best_Q, confidence_tol);
            double percent = double(pI.size())/double(Q.cols())*100;
            if (percent > params.less_confident_percent) {
                std::cout << "\n##CONV: Less confident pixels are greater than " << params.less_confident_percent 
                    << "%, terminating...\n";
                break;
            }
        }

    } while(it<maxiter);

#if VERBOSE
    int_energy = assignment_energy_true(currentMap(Q));
	std::cout <<"final projected energy: " << int_energy << "\n";

    // verify KKT conditions
    // gamma
    tmp = gamma.cwiseProduct(Q);
    std::cout << "gamma\t:: mean=" << gamma.mean() << ",\tmax=" << gamma.maxCoeff() << ",\tmin=" << gamma.minCoeff() << std::endl;
    std::cout << "gamma.cwiseProduct(Q)\t:: mean=" << tmp.mean() << ",\tmax=" << tmp.maxCoeff() << ",\tmin=" << tmp.minCoeff() << std::endl;
    std::cout << "beta_mat\t:: mean=" << beta_mat.mean() << ",\tmax=" << beta_mat.maxCoeff() << ",\tmin=" << beta_mat.minCoeff() << std::endl;
    std::cout << "alpha_tQ\t:: mean=" << alpha_tQ.mean() << ",\tmax=" << alpha_tQ.maxCoeff() << ",\tmin=" << alpha_tQ.minCoeff() << std::endl;
#endif

    return best_Q;
}

// LP inference with proximal algorithm with restricted pixel and labels- with tracing
std::vector<perf_measure> DenseCRF::tracing_lp_inference_prox_restricted(MatrixXf & init, LP_inf_params & params, double time_limit) const {

    MatrixXf best_Q(M_, N_), tmp(M_, N_), tmp2(M_, N_);
    MatrixP dot_tmp(M_, N_);
    MatrixXi ind(M_, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    MatrixXf unary = unary_->get();

    MatrixXf Q = init;
    renormalize(Q);
    assert(valid_probability_debug(Q));
    best_Q = Q;

    // Compute the value of the energy
    double energy = 0, best_energy = std::numeric_limits<double>::max(), 
		   best_int_energy = std::numeric_limits<double>::max();
    double int_energy = assignment_energy_true(currentMap(Q));
	best_int_energy = int_energy;
#if VERBOSE    
    double kl = klDivergence(Q, max_rounding(Q));
	energy = compute_energy_LP(Q);
	best_energy = energy;
    printf("Initial energy in the LP: %10.3f / %10.3f / %10.3f\n", energy, int_energy, kl);
#endif

	const int maxiter = params.prox_max_iter;
    const bool best_int = params.best_int;
    const bool accel_prox = params.accel_prox;
	const float prox_tol = params.prox_energy_tol;		// proximal energy tolerance
    const float confidence_tol = params.confidence_tol;

	// dual Frank-Wolfe variables
	const float dual_gap_tol = params.dual_gap_tol;		// dual gap tolerance
	const float lambda = params.prox_reg_const;	// proximal-regularization constant
	const int fw_maxiter = params.fw_max_iter;
	float delta = 1;						// FW step size
	double dual_gap = 0, dual_energy = 0;

    // gamma-qp variables
    const float qp_delta = params.qp_const;	// constant used in qp-gamma
    const float qp_tol = params.qp_tol;		// qp-gamma tolernace
    const int qp_maxiter = params.qp_max_iter;

    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;
    
    std::vector<int> pI;    // restricted pixels

    MatrixXf cur_Q(M_, N_);		// current Q in prox step
#if VERBOSE
  	MatrixXf int_Q(M_, N_);		// store integral Q
#endif
    // accelerated prox_lp
    MatrixXf prev_Q(M_, N_);	// Q resulted in prev prox step
    prev_Q.fill(0);
    float w_it = 1;             // momentum weight: eg. it/(it+3)

    int it=0;
    int count = 0;
    do {
        ++it;

        start = std::chrono::high_resolution_clock::now();

        // matrix creations
        less_confident_pixels(pI, Q, confidence_tol);
        //float percent = 10;
        //less_confident_pixels2(pI, Q, percent);
        int rN = pI.size();
        double percent = double(rN)/double(Q.cols())*100;
        if (percent < 1.0) {
            std::cout << "#CONV: Less confident pixels are less than 1%, exiting...\n";
            break;
        }
#if VERBOSE    
        std::cout << "No of pixels with probability less than " << confidence_tol << " is: " << pI.size() 
            << " out of " << Q.cols() << ", percentage: " << double(pI.size())/double(Q.cols())*100 << "%" << std::endl;
#endif
        // dual variables
        MatrixXf alpha_tQ(M_, rN);	// A * alpha, (t - tilde not iteration)
        MatrixXf s_tQ(M_, rN);		// A * s, conditional gradient of FW == subgradient
        VectorXf beta(rN);			// unconstrained --> correct beta values (beta.row(i) == v_beta forall i)
        MatrixXf beta_mat(M_, rN);	// beta_mat.row(i) == beta forall i --> N_ * M_ elements 
        MatrixXf gamma(M_, rN);		// nonnegative
        
        MatrixXf cur_rQ(M_, rN);		// current Q in prox step
        MatrixXf rescaled_rQ(M_, rN);   // infeasible Q rescaled to be within [0,1]
        MatrixXf rtmp(M_, rN), rtmp2(M_, rN);
        MatrixP rdot_tmp(M_, rN);
    
        MatrixXf rQ(M_, rN);		    // restricted Q in prox step
        update_restricted_matrix(rQ, Q, pI);
        MatrixXf runary(M_, rN);        // restricted unary
        update_restricted_matrix(runary, unary, pI);
        
	    //MatrixXf C(M_, M_), neg_C(M_, M_), pos_C(M_, M_), abs_C(M_, M_);
        VectorXf v_gamma(M_), v_y(M_), v_pos_h(M_), v_neg_h(M_), v_step(M_), v_tmp(M_), v_tmp2(M_);
        MatrixXf Y(M_, rN), neg_H(M_, rN), pos_H(M_, rN);
        VectorXf qp_values(rN);

//        pos_C = MatrixXf::Identity(M_, M_) * (1-1.0/float(M_));				      
//        pos_C *= lambda;	// pos_C
//        neg_C = MatrixXf::Ones(M_, M_) - MatrixXf::Identity(M_, M_);	
//        neg_C /= float(M_);													      
//        neg_C *= lambda; 	// neg_C
//        C = pos_C - neg_C;	// C
//        abs_C = pos_C + neg_C;	// abs_C

		// initialization
		cur_Q = Q;
        if (accel_prox) {   // accelerated prox
            w_it = float(it)/(it+3.0);  // simplest choice
            tmp = Q - prev_Q;
            tmp *= w_it;
            cur_Q += tmp;   // cur_Q = Q + w_it(Q - prev_Q)
            prev_Q = Q;
        }
        
		int pit = 0;
		alpha_tQ.fill(0);	// all zero alpha_tQ is feasible --> alpha^1_{abi} = alpha^2_{abi} = K_{ab}/4
		beta_mat.fill(0);
		beta.fill(0);
		gamma.fill(1);	// all zero is a fixed point of QP iteration!
        update_restricted_matrix(cur_rQ, cur_Q, pI);
		
		// prox step
		do {	
			++pit;
			// initialization
			s_tQ.fill(0);

            //if (pit == 1) { // only compute beta and gamma in the first iteration
			// QP-gamma -- \cite{NNQP solver Xiao and Chen 2014}
			// case-1: solve for gamma using qp solver! 
			// 1/2 * gamma^T * C * gamma - gamma^T * H
			// populate Y matrix
			rtmp = alpha_tQ - runary;
			for (int i = 0; i < rN; ++i) {
//                v_y = C * rtmp.col(i);
                // do it in linear time
                v_tmp2 = rtmp.col(i);
                qp_gamma_multiplyC(v_y, v_tmp2, M_, lambda);
				Y.col(i) = v_y;
			}	
			Y += cur_rQ;		// H = -Y
			for (int i = 0; i < rN; ++i) {
				for (int j = 0; j < M_; ++j) {	
					pos_H(j, i) = std::max(-Y(j, i), (float)0);		// pos_H 
					neg_H(j, i) = std::max(Y(j, i), (float)0);		// neg_H
				}
			}
			// qp iterations, 
			int qpit = 0;
			qp_values.fill(0);
		    //gamma.fill(1);	// initializing gamma here affects efficiency!
			float qp_value = std::numeric_limits<float>::max();
#if VERBOSE
            htime st = std::chrono::high_resolution_clock::now();
#endif
			do {
				//solve for each pixel separately
				for (int i = 0; i < rN; ++i) {
					v_gamma = gamma.col(i);
					v_pos_h = pos_H.col(i);
					v_neg_h = neg_H.col(i);
                    //
//                    v_step = 2 * neg_C * v_gamma + v_pos_h;
//					v_step = v_step.array() + qp_delta;
//					v_tmp = abs_C * v_gamma + v_neg_h;
//				    v_tmp = v_tmp.array() + qp_delta;
//					v_step = v_step.cwiseQuotient(v_tmp);
//					v_gamma = v_gamma.cwiseProduct(v_step);
                    // do it in linear time
                    qp_gamma_step(v_gamma, v_pos_h, v_neg_h, qp_delta, M_, lambda, v_step, v_tmp, v_tmp2);
					gamma.col(i) = v_gamma;

					// qp value
//                    v_tmp = C * v_gamma;
                    // do it in linear time
                    qp_gamma_multiplyC(v_tmp, v_gamma, M_, lambda);
					v_y = Y.col(i);
					qp_values(i) = 0.5 * v_gamma.dot(v_tmp) + v_gamma.dot(v_y);
				}
				float qp_value1 = qp_values.sum();
				//printf("\n#QP: %4d, %10.3f", qpit, qp_value1);
				++qpit;
				if (std::abs(qp_value - qp_value1) < qp_tol) break;
				qp_value = qp_value1;
			} while (qpit < qp_maxiter);
#if VERBOSE
            htime et = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
            printf("# Time-QP %d: %5.5f, %10.3f\t", qpit, dt, qp_value);
#endif
			// end-qp-gamma

			// case-2: update beta -- gradient of dual wrt beta equals to zero
			beta_mat = (alpha_tQ + gamma - runary);	// -B^T/l * (A * alpha + gamma - phi)
			// DON'T DO IT AT ONCE!! (RETURNS A SCALAR???)--> do it in two steps
			beta = -beta_mat.colwise().sum();	
			beta /= M_;
			// repeat beta in each row of beta_mat - - B * beta
			for (int j = 0; j < M_; ++j) {
				beta_mat.row(j) = beta;
			}
            //}

			// case-3: dual conditional-gradient or primal-subgradient (do it as the final case)
			rQ = lambda * (alpha_tQ + beta_mat + gamma - runary) + cur_rQ;	// Q may be infeasible --> but no problem

			// new PH implementation doesn't work with Q values outside [0,1] - or in fact truncates to be within [0,1]
    		// rescale Q to be within [0,1] -- order of Q values preserved!
    		rescale(rescaled_rQ, rQ);

			// subgradient lower minus upper
			// Pairwise
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
#if VERBOSE
                htime st = std::chrono::high_resolution_clock::now();
#endif
				// new PH implementation
				// rescaled_Q values in the range [0,1] --> but the same order as Q! --> subgradient of Q
                bool store = (pit == 1); // store only at the first iteration
                pairwise_[k]->apply_upper_minus_lower_ord_restricted(rtmp, rescaled_rQ, pI, Q, store);	

                s_tQ += rtmp;	// A * s is lower minus upper, keep neg introduced by compatibility->apply
#if VERBOSE
                htime et = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
                printf("# Time-%d: %5.5f\t", k, dt);
#endif
            }
			// find dual gap
			rtmp = alpha_tQ - s_tQ;	
			dual_gap = dotProduct(rtmp, rQ, rdot_tmp);

#if VERBOSE
			// dual-energy value
			rtmp2 = rQ - cur_rQ;
			dual_energy = dotProduct(rtmp2, rtmp2, rdot_tmp) / (2* lambda) + dotProduct(rtmp2, cur_rQ, rdot_tmp) / lambda;
			dual_energy -= beta.sum();
			double primal_energy = dotProduct(rtmp2, rtmp2, rdot_tmp) / (2* lambda);
		   	primal_energy += dotProduct(runary, rQ, rdot_tmp);
			primal_energy -= dotProduct(s_tQ, rQ, rdot_tmp);	// cancel the neg in s_tQ
			//assert(dual_gap == (dual_energy - primal_energy));
			printf("%4d: [%10.3f = %10.3f, %10.3f, %10.3f, ", pit-1, dual_gap, primal_energy+dual_energy, 
                    -dual_energy, primal_energy);
#endif
			if (dual_gap <= dual_gap_tol) break;	// stopping condition

			// optimal fw step size
			delta = (float)(dual_gap / (lambda * dotProduct(rtmp, rtmp, rdot_tmp)));
			delta = std::min(std::max(delta, (float)0.0), (float)1.0);  // I may not need to truncate the step-size!!
			assert(delta >= 0);
#if VERBOSE
			printf("%1.10f]\n", delta);
#endif
			// update alpha_tQ
			rtmp = s_tQ - alpha_tQ;
			rtmp *= delta;
			alpha_tQ += rtmp;	// alpha_tQ = alpha_tQ + delta * (s_tQ - alpha_tQ);

		} while(pit<fw_maxiter);

        update_extended_matrix(Q, rQ, pI);
		// project Q back to feasible region
		feasible_Q(tmp, ind, sum, K, Q);
		Q = tmp;
		renormalize(Q);
        assert(valid_probability_debug(Q));

        double prev_int_energy = int_energy;
        int_energy = assignment_energy_true(currentMap(Q));
#if VERBOSE
        double prev_energy = energy;
		energy = compute_energy_LP(Q);
//#if VERBOSE
        int_Q = max_rounding(Q);
        kl = klDivergence(Q, int_Q);
//#endif
#endif
        if (best_int) {
            if ((best_int_energy - int_energy) < prox_tol) ++count; // may also increase!
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: best_int_energy - int_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
		    if(int_energy < best_int_energy) {
                best_Q = Q;
                best_int_energy = int_energy;
            }
#if VERBOSE
            printf("%4d: %10.3f / %10.3f / %10.3f / %10.3f\n", it-1, energy, int_energy, kl, best_int_energy);
#endif
        } else {
#if VERBOSE == false
            double prev_energy = energy;
    		energy = compute_energy_LP(Q);
#endif
            if (abs(energy - prev_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: energy - prev_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
    		if( energy < best_energy) {
                best_Q = Q;
                best_energy = energy;
            }
#if VERBOSE
            printf("%4d: %10.3f / %10.3f / %10.3f / %10.3f\n", it-1, energy, int_energy, kl, best_energy);
#endif
        }

        end = std::chrono::high_resolution_clock::now();
        perf_timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
        perf_energy = best_int_energy;
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;

    } while(it < maxiter && (time_limit == 0 || total_time < time_limit));

    init = best_Q;
    return perfs;
}

// LP inference with proximal algorithm - tracing 
std::vector<perf_measure> DenseCRF::tracing_lp_inference_prox(MatrixXf & init, LP_inf_params & params, 
        double time_limit, std::string out_file_name) const {

    MatrixXf best_Q(M_, N_), tmp(M_, N_), tmp2(M_, N_);
    MatrixP dot_tmp(M_, N_);
    MatrixXi ind(M_, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    MatrixXf unary = unary_->get();

    MatrixXf Q = init;
    renormalize(Q);
    assert(valid_probability_debug(Q));
    best_Q = Q;

    bool dump = false;
    std::ofstream fout;
    if (out_file_name != "") {
        dump = true;
        fout.open(out_file_name.c_str());
    }

    // Compute the value of the energy
    double energy = 0, best_energy = std::numeric_limits<double>::max(), 
		   best_int_energy = std::numeric_limits<double>::max();
    double int_energy = assignment_energy_true(currentMap(Q));
#if VERBOSE    
    double kl = klDivergence(Q, max_rounding(Q));
#endif
	energy = compute_energy_LP(Q);
    if (energy > int_energy) {  // choose the best initialization 
        Q = max_rounding(Q);
    	energy = compute_energy_LP(Q);
    }
	best_energy = energy;
	best_int_energy = int_energy;
#if VERBOSE    
    std::cout << "#file: " << out_file_name << ", dump: " << dump << std::endl;
    printf("Initial energy in the LP: %10.3f / %10.3f / %10.3f", energy, int_energy, kl);
    if (dump) fout << "Initial energy in the LP-Prox: " << energy << " / " << int_energy << " / " << kl;
#endif

	const int maxiter = params.prox_max_iter;
    const bool best_int = params.best_int;
    const bool accel_prox = params.accel_prox;
	const float prox_tol = params.prox_energy_tol;		// proximal energy tolerance

	// dual Frank-Wolfe variables
	const float dual_gap_tol = params.dual_gap_tol;		// dual gap tolerance
	const float lambda = params.prox_reg_const;	// proximal-regularization constant
	const int fw_maxiter = params.fw_max_iter;
	float delta = 1;						// FW step size
	double dual_gap = 0, dual_energy = 0;

	// dual variables
	MatrixXf alpha_tQ(M_, N_);	// A * alpha, (t - tilde not iteration)
	MatrixXf s_tQ(M_, N_);		// A * s, conditional gradient of FW == subgradient
	VectorXf beta(N_);			// unconstrained --> correct beta values (beta.row(i) == v_beta forall i)
	MatrixXf beta_mat(M_, N_);	// beta_mat.row(i) == beta forall i --> N_ * M_ elements 
	MatrixXf gamma(M_, N_);		// nonnegative
	
	MatrixXf cur_Q(M_, N_);		// current Q in prox step
	MatrixXf rescaled_Q(M_, N_);// infeasible Q rescaled to be within [0,1]
#if VERBOSE
	MatrixXf int_Q(M_, N_);		// store integral Q
#endif

    // accelerated prox_lp
	MatrixXf prev_Q(M_, N_);	// Q resulted in prev prox step
    prev_Q.fill(0);
    float w_it = 1;             // momentum weight: eg. it/(it+3)

	// gamma-qp variables
	const float qp_delta = params.qp_const;	// constant used in qp-gamma
	const float qp_tol = params.qp_tol;		// qp-gamma tolernace
	const int qp_maxiter = params.qp_max_iter;

	//MatrixXf C(M_, M_), neg_C(M_, M_), pos_C(M_, M_), abs_C(M_, M_);
	VectorXf v_gamma(M_), v_y(M_), v_pos_h(M_), v_neg_h(M_), v_step(M_), v_tmp(M_), v_tmp2(M_);
	MatrixXf Y(M_, N_), neg_H(M_, N_), pos_H(M_, N_);
	VectorXf qp_values(N_);

//	pos_C = MatrixXf::Identity(M_, M_) * (1-1.0/M_);				      
//	pos_C *= lambda;	// pos_C
//	neg_C = MatrixXf::Ones(M_, M_) - MatrixXf::Identity(M_, M_);	
//	neg_C /= M_;													      
//	neg_C *= lambda; 	// neg_C
//	C = pos_C - neg_C;	// C
//	abs_C = pos_C + neg_C;	// abs_C
    
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;
    //clock_t start, end;
    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    int it=0;
    int count = 0;
    do {
        bool stop = false;
        ++it;

        //start = clock();
        start = std::chrono::high_resolution_clock::now();

		// initialization
		beta_mat.fill(0);
		beta.fill(0);
		gamma.fill(1);	// all zero is a fixed point of QP iteration!
		cur_Q = Q;
        
        if (accel_prox) {   // accelerated prox
            w_it = float(it)/(it+3.0);  // simplest choice
            tmp = Q - prev_Q;
            tmp *= w_it;
            cur_Q += tmp;   // cur_Q = Q + w_it(Q - prev_Q)
            prev_Q = Q;
        }
        
		int pit = 0;
		alpha_tQ.fill(0);	// all zero alpha_tQ is feasible --> alpha^1_{abi} = alpha^2_{abi} = K_{ab}/4
		
		// prox step
		do {	
			++pit;
			// initialization
			s_tQ.fill(0);

            //if (pit == 1) { // only compute beta and gamma in the first iteration
			// QP-gamma -- \cite{NNQP solver Xiao and Chen 2014}
			// case-1: solve for gamma using qp solver! 
			// 1/2 * gamma^T * C * gamma - gamma^T * H
			// populate Y matrix
			tmp = alpha_tQ - unary;
			for (int i = 0; i < N_; ++i) {
//				v_y = C * tmp.col(i);
                // do it in linear time
                v_tmp2 = tmp.col(i);
                qp_gamma_multiplyC(v_y, v_tmp2, M_, lambda);
				Y.col(i) = v_y;
			}	
			Y += cur_Q;		// H = -Y
			for (int i = 0; i < N_; ++i) {
				for (int j = 0; j < M_; ++j) {	
					pos_H(j, i) = std::max(-Y(j, i), (float)0);		// pos_H 
					neg_H(j, i) = std::max(Y(j, i), (float)0);		// neg_H
				}
			}
			// qp iterations, 
			int qpit = 0;
			qp_values.fill(0);
		    //gamma.fill(1);	// initializing gamma here affects efficiency!
			float qp_value = std::numeric_limits<float>::max();
#if VERBOSE
            //clock_t st = clock();
            htime st = std::chrono::high_resolution_clock::now();
#endif
			do {
				//solve for each pixel separately
				for (int i = 0; i < N_; ++i) {
					v_gamma = gamma.col(i);
					v_pos_h = pos_H.col(i);
					v_neg_h = neg_H.col(i);
					//
//                    v_step = 2 * neg_C * v_gamma + v_pos_h;
//					v_step = v_step.array() + qp_delta;
//					v_tmp = abs_C * v_gamma + v_neg_h;
//				    v_tmp = v_tmp.array() + qp_delta;
//					v_step = v_step.cwiseQuotient(v_tmp);
//					v_gamma = v_gamma.cwiseProduct(v_step);
                    // do it linear time
                    qp_gamma_step(v_gamma, v_pos_h, v_neg_h, qp_delta, M_, lambda, v_step, v_tmp, v_tmp2);
					gamma.col(i) = v_gamma;

					// qp value
//                    v_tmp = C * v_gamma;
                    // do it in linear time
                    qp_gamma_multiplyC(v_tmp, v_gamma, M_, lambda);
					v_y = Y.col(i);
					qp_values(i) = 0.5 * v_gamma.dot(v_tmp) + v_gamma.dot(v_y);
				}
				float qp_value1 = qp_values.sum();
				//printf("\n#QP: %4d, %10.3f", qpit, qp_value1);
				++qpit;
				if (std::abs(qp_value - qp_value1) < qp_tol) break;
				qp_value = qp_value1;
			} while (qpit < qp_maxiter);
#if VERBOSE
            //clock_t et = clock();
            htime et = std::chrono::high_resolution_clock::now();
            //double dt = (double)(et-st)/CLOCKS_PER_SEC;
            double dt = std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
            printf("\n# Time-QP %d: %5.5f, %10.3f\t", qpit, dt, qp_value);
            if (dump) fout << "\n# Time-QP " << qpit << ": " << dt << ", " << qp_value << '\t';
#endif
			// end-qp-gamma

			// case-2: update beta -- gradient of dual wrt beta equals to zero
			beta_mat = (alpha_tQ + gamma - unary);	// -B^T/l * (A * alpha + gamma - phi)
			// DON'T DO IT AT ONCE!! (RETURNS A SCALAR???)--> do it in two steps
			beta = -beta_mat.colwise().sum();	
			beta /= M_;
			// repeat beta in each row of beta_mat - - B * beta
			for (int j = 0; j < M_; ++j) {
				beta_mat.row(j) = beta;
			}
            //}

			// case-3: dual conditional-gradient or primal-subgradient (do it as the final case)
			Q = lambda * (alpha_tQ + beta_mat + gamma - unary) + cur_Q;	// Q may be infeasible --> but no problem

			// new PH implementation doesn't work with Q values outside [0,1] - or in fact truncates to be within [0,1]
    		// rescale Q to be within [0,1] -- order of Q values preserved!
    		rescale(rescaled_Q, Q);

			// subgradient lower minus upper
			// Pairwise
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
#if VERBOSE
                //st = clock();
                st = std::chrono::high_resolution_clock::now();
#endif
				// new PH implementation
				// rescaled_Q values in the range [0,1] --> but the same order as Q! --> subgradient of Q
                pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_Q);	

                s_tQ += tmp;	// A * s is lower minus upper, keep neg introduced by compatibility->apply
#if VERBOSE
                //et = clock();
                et = std::chrono::high_resolution_clock::now();
                //dt = (double)(et-st)/CLOCKS_PER_SEC;
                dt = std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
                printf("# Time-%d: %5.5f\t", k, dt);
                if (dump) fout << "# Time-" << k << ": " << dt << '\t';
#endif
            }
			// find dual gap
			tmp = alpha_tQ - s_tQ;	
			dual_gap = dotProduct(tmp, Q, dot_tmp);

#if VERBOSE
			// dual-energy value
			tmp2 = Q - cur_Q;
			dual_energy = dotProduct(tmp2, tmp2, dot_tmp) / (2* lambda) + dotProduct(tmp2, cur_Q, dot_tmp) / lambda;
			dual_energy -= beta.sum();
			double primal_energy = dotProduct(tmp2, tmp2, dot_tmp) / (2* lambda);
		   	primal_energy += dotProduct(unary, Q, dot_tmp);
			primal_energy -= dotProduct(s_tQ, Q, dot_tmp);	// cancel the neg in s_tQ
			//assert(dual_gap == (dual_energy - primal_energy));
			printf("%4d: [%10.3f = %10.3f, %10.3f, %10.3f, ", pit-1, dual_gap, primal_energy+dual_energy, 
                    -dual_energy, primal_energy);
			if (dump) fout << pit-1 << ": [" << dual_gap << " = " << primal_energy+dual_energy << ", "
                    << -dual_energy << ", " << primal_energy << ", ";
            //if (dual_gap < 0) {   // may become negative due to PH approximations!
            //    std::cout << "\nERROR: Dual-gap cannot be negative!\n";
            //    exit(1);
            //}
#endif
			if (dual_gap <= dual_gap_tol) break;	// stopping condition

			// optimal fw step size
			delta = (float)(dual_gap / (lambda * dotProduct(tmp, tmp, dot_tmp)));
			delta = std::min(std::max(delta, (float)0.0), (float)1.0);  // I may not need to truncate the step-size!!
			assert(delta > 0);
#if VERBOSE
			printf("%1.10f]", delta);
			if (dump) fout << delta << "]";
#endif
			// update alpha_tQ
			tmp = s_tQ - alpha_tQ;
			tmp *= delta;
			alpha_tQ += tmp;	// alpha_tQ = alpha_tQ + delta * (s_tQ - alpha_tQ);

		} while(pit<fw_maxiter);

		// project Q back to feasible region
		feasible_Q(tmp, ind, sum, K, Q);
		Q = tmp;
		renormalize(Q);
        assert(valid_probability_debug(Q));


        double prev_int_energy = int_energy;
        int_energy = assignment_energy_true(currentMap(Q));
#if VERBOSE
        double prev_energy = energy;
		energy = compute_energy_LP(Q);
        int_energy = assignment_energy_true(currentMap(Q));
        int_Q = max_rounding(Q);
        kl = klDivergence(Q, int_Q);
#endif

        if (best_int) {
            if (abs(int_energy - prev_int_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: int_energy - prev_int_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                if (dump) fout << "\n##CONV: int_energy - prev_int_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
		    if(int_energy < best_int_energy) {
                best_Q = Q;
                best_int_energy = int_energy;
            }
#if VERBOSE
            printf("\n# Prox%4d: %10.3f / %10.3f / %10.3f / %10.3f [%4d, %10.3f]\n", it-1, energy, 
                int_energy, kl, best_int_energy, pit-1, dual_gap);
            if (dump) fout << "\n# Prox-" << it-1 << ": " << energy << " / " << int_energy << " / " << kl << " / " 
                << best_int_energy << " [" << pit-1 << ", " << dual_gap << "]" << std::endl;
#endif
        } else {
#if VERBOSE == false
            double prev_energy = energy;
    		energy = compute_energy_LP(Q);
#endif
            if (abs(energy - prev_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                std::cout << "\n##CONV: energy - prev_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                if (dump) fout << "\n##CONV: energy - prev_energy < " << prox_tol << " for last " << count 
                    << " iterations! terminating...\n";
                break;
            }
    		if( energy < best_energy) {
                best_Q = Q;
                best_energy = energy;
            }
#if VERBOSE
            printf("\n%4d: %10.3f / %10.3f / %10.3f / %10.3f [%4d, %10.3f]\n", it-1, energy, int_energy, 
                kl, best_energy, pit-1, dual_gap);
            if (dump) fout << "\n" << it-1 << ": " << energy << " / " << int_energy << " / " << kl << " / " 
                << best_energy << " [" << pit-1 << ", " << dual_gap << "]" << std::endl;
#endif
        }
        if (params.less_confident_percent > 0) {
            float confidence_tol = params.confidence_tol;
            std::vector<int> pI;
            less_confident_pixels(pI, best_Q, confidence_tol);
            double percent = double(pI.size())/double(Q.cols())*100;
            if (percent > params.less_confident_percent) {
                std::cout << "\n##CONV: Less confident pixels are greater than " << params.less_confident_percent 
                    << "%, terminating...\n";
                if (dump) fout << "\n##CONV: Less confident pixels are greater than " << params.less_confident_percent 
                    << "%, terminating...\n";
                //break;
                stop = true;
            }
        }

        //end = clock();
        //perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        end = std::chrono::high_resolution_clock::now();
        perf_timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
        perf_energy = best_int_energy;
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;

        if (stop) break;

    } while(it < maxiter && (time_limit == 0 || total_time < time_limit));

    if (dump) fout.close();

    init = best_Q;
    return perfs;
}

std::vector<perf_measure> DenseCRF::tracing_lp_inference(MatrixXf & init, bool use_cond_grad, double time_limit, bool full_mat) const {
    // Restrict number of labels in the computation
    std::vector<int> indices;
    if (full_mat) { // hack
        for(int i = 0; i < M_; ++i) indices.push_back(i);
    } else {
        get_limited_indices(init, indices);
    }
    int restricted_M = indices.size();
    MatrixXf unary = get_restricted_matrix(unary_->get(), indices);
    MatrixXf Q = get_restricted_matrix(init, indices);
    renormalize(Q);

    clock_t start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;

    // std::cout<<"Using only "<<indices.size()<<" labels"<<std::endl;


    MatrixXf best_Q(restricted_M, N_), ones(restricted_M, N_), base_grad(restricted_M, N_), tmp(restricted_M, N_),
        grad(restricted_M, N_), tmp2(restricted_M, N_), desc(restricted_M, N_), tmp_single_line(1, N_);
    MatrixP dot_tmp(restricted_M, N_);
    MatrixXi ind(restricted_M, N_);
    VectorXi K(N_);
    VectorXd sum(N_);

    // Create copies of the original pairwise since we don't want normalization
    int nb_pairwise = pairwise_.size();
    PairwisePotential** no_norm_pairwise;
    no_norm_pairwise = (PairwisePotential**) malloc(pairwise_.size()*sizeof(PairwisePotential*));
    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        no_norm_pairwise[k] = new PairwisePotential(
            pairwise_[k]->features(),
            new PottsCompatibility(pairwise_[k]->parameters()(0)),
            pairwise_[k]->ktype(),
            NO_NORMALIZATION
            );
    }

    ones.fill(1);

    int i,j;

    best_Q = Q;
    double best_int_energy = assignment_energy_true(get_original_label(currentMap(Q), indices));

    // Compute the value of the energy
    double old_energy;
    assert(valid_probability(Q));
    double energy = 0;

    int it=0;
    do {
        start = clock();
        ++it;
        old_energy = energy;

        // Compute the current energy and gradient
        // Unary
        energy = dotProduct(unary, Q, dot_tmp);
        grad = unary;

        // Pairwise
        sortRows(Q, ind);
        for( unsigned int k=0; k<nb_pairwise; k++ ) {
            // Special case for 2 labels (used mainly for alpha expansion)
            if(Q.rows()==2) {
                // Compute only for one label
                no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp_single_line, ind.row(0));
                tmp2.fill(0);
                for(i=0; i<tmp.cols(); ++i) {
                    tmp2(0, ind(0, i)) = tmp_single_line(i);
                }
                // Closed form solution for the other label
                tmp2.row(1) = - tmp2.row(0);
            } else {
                // Add upper minus lower
                no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp, ind);
                tmp2.fill(0);
                for(i=0; i<tmp.cols(); ++i) {
                    for(j=0; j<tmp.rows(); ++j) {
                        tmp2(j, ind(j, i)) = tmp(j, i);
                    }
                }
            }

            energy -= dotProduct(Q, tmp2, dot_tmp);
            grad -= tmp2;
        }
        // Print previous iteration energy
        // std::cout << it << ": " << energy << "\n";

        // Sub-gradient descent step
        double int_energy = 0;
        if(use_cond_grad) {
            descent_direction(desc, grad);

            float min = 0.0;
            float max = 1.0;
            double min_int_energy, max_int_energy, left_third_int_energy, right_third_int_energy;
            int split = 0;
            min_int_energy = assignment_energy_true(get_original_label(currentMap(Q), indices));
            max_int_energy = assignment_energy_true(get_original_label(currentMap(desc), indices));
            do {
                split++;
                double left_third = (2*min + max)/3.0;
                double right_third = (min + 2*max)/3.0;
                left_third_int_energy = assignment_energy_true(get_original_label(currentMap(Q+(desc-Q)*left_third), indices));
                right_third_int_energy = assignment_energy_true(get_original_label(currentMap(Q+(desc-Q)*right_third), indices));
                if(left_third_int_energy < right_third_int_energy) {
                    max = right_third;
                    max_int_energy = right_third_int_energy;
                    int_energy = left_third_int_energy;
                } else {
                    min = left_third;
                    min_int_energy = left_third_int_energy;
                    int_energy = right_third_int_energy;
                }
            } while(max-min > 0.001);
            //std::cout<<" learning rate: "<<(max+min)/2.0<<" expected: "<<min_int_energy<<std::endl;

            Q += 0.5*(max+min)*(desc - Q);
        } else {
            float min = 0.0;
            float max = 1e-3;
            double min_int_energy, max_int_energy, left_third_int_energy, right_third_int_energy;
            int split = 0;
            min_int_energy = assignment_energy_true(get_original_label(currentMap(Q), indices));
            max_int_energy = assignment_energy_true(get_original_label(currentMap(Q-max*grad), indices));
            do {
                split++;
                double left_third = (2*min + max)/3.0;
                double right_third = (min + 2*max)/3.0;
                left_third_int_energy = assignment_energy_true(get_original_label(currentMap(Q-left_third*grad), indices));
                right_third_int_energy = assignment_energy_true(get_original_label(currentMap(Q-right_third*grad), indices));
                if(left_third_int_energy < right_third_int_energy) {
                    max = right_third;
                    max_int_energy = right_third_int_energy;
                    int_energy = left_third_int_energy;
                } else {
                    min = left_third;
                    min_int_energy = left_third_int_energy;
                    int_energy = right_third_int_energy;
                }
            } while(max-min > 0.00001);

            Q -= 0.5*(max+min)*grad;

            // Project current estimates on valid space
            sortCols(Q, ind);
            for(int i=0; i<N_; ++i) {
                sum(i) = Q.col(i).sum()-1;
                K(i) = -1;
            }
            for(int i=0; i<N_; ++i) {
                for(int k=restricted_M; k>0; --k) {
                    double uk = Q(ind(k-1, i), i);
                    if(sum(i)/k < uk) {
                        K(i) = k;
                        break;
                    }
                    sum(i) -= uk;
                }
            }
            tmp.fill(0);
            for(int i=0; i<N_; ++i) {
                for(int k=0; k<restricted_M; ++k) {
                    tmp(k, i) = std::max(Q(k, i) - sum(i)/K(i), (double)0);
                }
            }
            Q = tmp;
        }

        if(int_energy < best_int_energy) {
            best_Q = Q;
            best_int_energy = int_energy;
        }

        end = clock();
        perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        perf_energy = best_int_energy;
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;
        if (time_limit != 0 and total_time>time_limit) {
            break;
        }

        assert(valid_probability(Q));
    } while(it<5 or time_limit != 0);
    // std::cout <<"final projected energy: " << best_int_energy << "\n";
    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        delete no_norm_pairwise[k];
    }
    free(no_norm_pairwise);

    // Reconstruct an output with the correct number of labels
    best_Q = get_extended_matrix(best_Q, indices, M_);
    init = best_Q;
    return perfs;
}

// only calculate pairwise energies 
void DenseCRF::compare_energies(MatrixXf & Q, double & ph_energy, double & bf_energy, 
		bool qp, bool ph_old, bool subgrad) const {
//	if (pairwise_.size() != 1) {
//		std::cout << "pairwise-size: " << pairwise_.size() << " (should be 1)" << std::endl;
//		exit(1);
//	}
    renormalize(Q);
	if (!valid_probability(Q)) {
		std::cout << "Q is not a valid probability!" << std::endl;
		exit(1);
	}
	
	// for bruteforce computation
	// Create copies of the original pairwise since we don't want normalization
    PairwisePotential** no_norm_pairwise;
    no_norm_pairwise = (PairwisePotential**) malloc(pairwise_.size()*sizeof(PairwisePotential*));
    for (int k = 0; k < pairwise_.size(); ++k) {
        no_norm_pairwise[k] = new PairwisePotential(
            pairwise_[k]->features(),
            new PottsCompatibility(pairwise_[k]->parameters()(0)),
            pairwise_[k]->ktype(),
            NO_NORMALIZATION
            );
    }
	
    MatrixXf tmp(M_, N_), tmp2(M_, N_), rescaled_Q(M_, N_);
	MatrixXi ind(M_, N_);
	MatrixP dot_tmp;
	double energy = 0;
    MatrixXf ph_grad(M_, N_), bf_grad(M_, N_);
    ph_grad.fill(0);
    bf_grad.fill(0);

	if (qp) {
		// ph-energy
		energy = 0;
        for (int k = 0; k < pairwise_.size(); ++k) {
            //pairwise_[k]->apply( tmp, Q );
            no_norm_pairwise[k]->apply( tmp, Q );   // must be no-normalized pairwise term
            if (subgrad) ph_grad -= tmp;
        	energy += dotProduct(Q, tmp, dot_tmp);	// do not cancel the neg intoduced in apply
        	// constant term
        	tmp = -tmp;	// cancel the neg introdcued in apply
        	tmp.transposeInPlace();
        	tmp2 = Q*tmp;	
        	double const_energy = tmp2.sum();
        	energy += const_energy;
        }
		ph_energy = energy;

		//bf-energy
		energy = 0;
        for (int k = 0; k < pairwise_.size(); ++k) {
    		no_norm_pairwise[k]->apply_bf( tmp, Q );
            if (subgrad) bf_grad -= tmp;
    		energy += dotProduct(Q, tmp, dot_tmp);	// do not cancel the neg intoduced in apply
    		// constant term
    		tmp = -tmp;	// cancel the neg introdcued in apply
    		tmp.transposeInPlace();
    		tmp2 = Q*tmp;	
    		double const_energy = tmp2.sum();
    		energy += const_energy;
        }
		bf_energy = energy;

	} else {
		// ph-energy
		energy = 0;
        if (ph_old) sortRows(Q, ind);
        else rescale(rescaled_Q, Q);
        for (int k = 0; k < pairwise_.size(); ++k) {
    		// old-ph
    		if (ph_old) {
                no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp2, ind);
        		// need to sort before dot-product
        		for(int i=0; i<tmp2.cols(); ++i) {
                	for(int j=0; j<tmp2.rows(); ++j) {
                    	tmp(j, ind(j, i)) = tmp2(j, i);
                	}
                }
    		} else {
            	// Add the upper minus the lower
    	        pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_Q);
    		}
    		//
            if (subgrad) ph_grad -= tmp;
            energy -= dotProduct(Q, tmp, dot_tmp);
        }
		ph_energy = energy;

		// bf-energy
		energy = 0;
		sortRows(Q, ind);
        for (int k = 0; k < pairwise_.size(); ++k) {
            if (ph_old) no_norm_pairwise[k]->apply_upper_minus_lower_bf(tmp2, ind);
            else no_norm_pairwise[k]->apply_upper_minus_lower_bf_ord(tmp2, ind, Q);
    		// need to sort before dot-product
    		for(int i=0; i<tmp2.cols(); ++i) {
            	for(int j=0; j<tmp2.rows(); ++j) {
                	tmp(j, ind(j, i)) = tmp2(j, i);
            	}
            }
            if (subgrad) bf_grad -= tmp;
            energy -= dotProduct(Q, tmp, dot_tmp);
        }
		bf_energy = energy;
	}

    if (subgrad) {  // compare subgradients
        // should be coliner 
        MatrixXf ph_bf = ph_grad - bf_grad;
        double costh = dotProduct(ph_grad, bf_grad, dot_tmp)/
            (sqrt(dotProduct(ph_grad, ph_grad, dot_tmp))*sqrt(dotProduct(bf_grad, bf_grad, dot_tmp)));
        std::cout << "#cos-theta: " << costh << std::endl;
        std::cout << "BF   :: mean=" << bf_grad.mean() << ",\tmax=" << bf_grad.maxCoeff() << ",\tmin=" << bf_grad.minCoeff() << std::endl;
        std::cout << "PH   :: mean=" << ph_grad.mean() << ",\tmax=" << ph_grad.maxCoeff() << ",\tmin=" << ph_grad.minCoeff() << std::endl;
        std::cout << "PH-BF:: mean=" << ph_bf.mean() << ",\tmax=" << ph_bf.maxCoeff() << ",\tmin=" << ph_bf.minCoeff() << std::endl;
    }
}

// compare lp-subgrad computation times
std::vector<perf_measure> DenseCRF::compare_lpsubgrad_timings(MatrixXf & Q, bool cmp_subgrad) const {
    // ensure strict ordering by adding noise smaller than RESOLUTION of new PH-lattice
    renormalize(Q);
    for (int i = 0; i < Q.cols(); ++i) {
        for (int j = 0; j < Q.rows(); ++j) {
            int maxint = Q.cols()*Q.rows()*10;
            int r = std::rand() % maxint;
            Q(j, i) += float(r)/(RESOLUTION * maxint);  // add noise of magnitude < 1/RESOLUTION
        }
    }

    renormalize(Q);
	if (!valid_probability(Q)) {
		std::cout << "Q is not a valid probability!" << std::endl;
		exit(1);
	}
	
    MatrixXf tmp(M_, N_), tmp2(M_, N_), rescaled_Q(M_, N_);
	MatrixXi ind(M_, N_);
	MatrixP dot_tmp;
    MatrixXf old_grad(M_, N_), new_grad(M_, N_);
    old_grad.fill(0);
    new_grad.fill(0);

    std::vector<perf_measure> perfs;

	// old-ph
    clock_t st, et;
    st = clock();
    sortRows(Q, ind);
    et = clock();
    double sort_time = (double)(et-st)/CLOCKS_PER_SEC;

    st = clock();
    rescale(rescaled_Q, Q);
    et = clock();
    double rescale_time = (double)(et-st)/CLOCKS_PER_SEC;

    for (int k = 0; k < pairwise_.size(); ++k) {
        st = clock();
        // old -ph
        no_norm_pairwise_[k]->apply_upper_minus_lower_dc(tmp2, ind);
    	// need to sort before dot-product
    	for(int i=0; i<tmp2.cols(); ++i) {
        	for(int j=0; j<tmp2.rows(); ++j) {
            	tmp(j, ind(j, i)) = tmp2(j, i);
        	}
        }
        et = clock();
        double old_timing = (double)(et-st)/CLOCKS_PER_SEC + sort_time;
        old_grad -= tmp;
        
        st = clock();
        // new ph
        pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_Q);
        et = clock();
        double new_timing = (double)(et-st)/CLOCKS_PER_SEC + rescale_time;
        new_grad -= tmp;

        perfs.push_back(std::make_pair(old_timing, new_timing));
    }
    	
    if (cmp_subgrad) {  // compare subgradients
        // should be coliner 
        MatrixXf diff = old_grad - new_grad;
        double costh = dotProduct(old_grad, new_grad, dot_tmp)/
            (sqrt(dotProduct(old_grad, old_grad, dot_tmp))*sqrt(dotProduct(new_grad, new_grad, dot_tmp)));
        std::cout << "#cos-theta: " << costh << std::endl;
        std::cout << "OLD   :: mean=" << old_grad.mean() << ",\tmax=" << old_grad.maxCoeff() << ",\tmin=" << old_grad.minCoeff() << std::endl;
        std::cout << "NEW   :: mean=" << new_grad.mean() << ",\tmax=" << new_grad.maxCoeff() << ",\tmin=" << new_grad.minCoeff() << std::endl;
        std::cout << "DIFF  :: mean=" << diff.mean() << ",\tmax=" << diff.maxCoeff() << ",\tmin=" << diff.minCoeff() << std::endl;

        double old_e = 0, new_e = 0;
        for (int k = 0; k < pairwise_.size(); ++k) {
            old_e += dotProduct(old_grad, Q, dot_tmp);
            new_e += dotProduct(new_grad, Q, dot_tmp);
        }
        std::cout << "old-energy: " << old_e << ", new-energy: " << new_e << std::endl;
    }
    return perfs;
}

MatrixXf DenseCRF::max_rounding(const MatrixXf &estimates) const {
    MatrixXf rounded = MatrixXf::Zero(estimates.rows(), estimates.cols());
    int argmax;
    for (int col=0; col<estimates.cols(); col++) {
        estimates.col(col).maxCoeff(&argmax);
        rounded(argmax, col) = 1;
    }
    return rounded;
}

MatrixXf DenseCRF::interval_rounding(const MatrixXf &estimates, int nb_random_rounding) const {
    MatrixXf best_rounded;
    double best_energy = 1e18;

    MatrixXf rounded;
    int assigned_labels;
    int to_assign = estimates.cols();
    double rounded_energy;

    for (int it=0; it<nb_random_rounding; it++) {
        rounded = MatrixXf::Zero(estimates.rows(), estimates.cols());
        assigned_labels = 0;
        std::vector<bool> assigned(to_assign, false);
        while (assigned_labels < to_assign) {
            int label_index = rand() % estimates.rows();
            float interval = (float) rand()/ RAND_MAX;

            for (int col=0; col < to_assign; col++) {
                if (not assigned[col]) { // check that we need to assign something
                    if (interval <= estimates(label_index, col)) { // check that this pixel should be assigned
                        assigned[col] = true;
                        assigned_labels++;
                        rounded(label_index, col) = 1;
                    }
                }
            }
        }
        rounded_energy = compute_energy(rounded);
        if (rounded_energy < best_energy) {
            best_energy = rounded_energy;
            best_rounded = rounded;
        }
    }
    return best_rounded;
}


MatrixXf DenseCRF::cccp_inference(const MatrixXf & init) const {
    MatrixXf Q( M_, N_), tmp1, unary(M_, N_), tmp2, old_Q(M_, N_);
    float old_kl, kl;
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    // Compute the largest eigenvalues
    float lambda_eig = 0;
    for (int i=0; i<pairwise_.size(); i++) {
        lambda_eig += pick_lambda_eig_to_concave(pairwise_[i]->compatibility_matrix(M_));
    }
    Q = init;

    bool keep_inferring = true;
    if (compute_kl) {
        old_kl = 0;
        kl = klDivergence(Q);
    }

    old_Q = Q;
    int count = 0;
    while(keep_inferring) {
        MatrixXf Cste = unary;
        Cste += MatrixXf::Ones(Q.rows(), Q.cols());
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp2, old_Q);
            Cste += tmp2;
        }

        Cste += -2* lambda_eig * old_Q;

        for(int var = 0; var < N_; ++var){
            VectorXf state(M_ + 1);
            state.head(M_) = old_Q.col(var);
            state(M_) = 1;

            newton_cccp(state, Cste.col(var), lambda_eig);
            Q.col(var) = state.head(M_);
        }
        if (compute_kl) {
            kl = klDivergence(Q);
            float kl_change = old_kl - kl;
            keep_inferring = (kl_change > 0.001);
            old_kl = kl;
        } else {
            float Q_change = (old_Q - Q).squaredNorm();
            keep_inferring = (Q_change > 0.001);
        }
        old_Q = Q;
        count++;
    }
    return Q;
}


MatrixXf DenseCRF::grad_inference(const MatrixXf & init) const {
    MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2, old_Q(M_, N_), Q_prev_lambda(M_, N_);
    unary.fill(0);
    if( unary_ ) {
        unary = unary_->get();
    }
    Q = init;

    bool keep_decreasing_lambda = true;
    float lambda = 1;
    Q_prev_lambda = Q;
    while(keep_decreasing_lambda){
        lambda /= 1.1;

        int count = 0;
        bool keep_inferring = true;
        old_Q = Q;
        while(keep_inferring) {
            tmp1 = -unary;
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
                pairwise_[k]->apply( tmp2, Q );
                tmp1 -= tmp2;
            }
            tmp1 = (1/lambda) * tmp1;
            expAndNormalize( Q, tmp1 );

            float Q_change = (old_Q - Q).squaredNorm();
            keep_inferring = (Q_change > 0.001);
            old_Q = Q;
            ++count;
        }

        float Q_lambda_change = (Q_prev_lambda - Q).squaredNorm();
        keep_decreasing_lambda = (Q_lambda_change > 0.001);
        Q_prev_lambda = Q;

    }

    return Q;
}


VectorXs DenseCRF::map ( int n_iterations ) const {
    // Run inference
    MatrixXf Q = inference(unary_init(),  n_iterations );
    // Find the map
    return currentMap( Q );
}

double DenseCRF::assignment_energy( const VectorXs & l) const {
    VectorXf unary = unaryEnergy(l);
    VectorXf pairwise = pairwiseEnergy(l);

    // Due to the weird way that the pairwise Energy is computed, this
    // is how we get results that correspond to what would be given by
    // binarizing the estimates, and using the compute_energy function.
    VectorXf total_energy = unary -2* pairwise;

    assert( total_energy.rows() == N_);
    double ass_energy = 0;
    for( int i=0; i< N_; ++i) {
        ass_energy += total_energy[i];
    }

    return ass_energy;
}

double DenseCRF::assignment_energy_true( const VectorXs & l) const {
    VectorXf unary = unaryEnergy(l);
    VectorXf pairwise = pairwise_energy_true(l);

    VectorXf total_energy = unary + pairwise;
 
    assert( total_energy.rows() == N_);
    double ass_energy = 0;
    for( int i=0; i< N_; ++i) {
        ass_energy += total_energy[i];
    }

    return ass_energy;
}

VectorXf DenseCRF::unaryEnergy(const VectorXs & l) const{
    assert( l.rows() == N_ );
    VectorXf r( N_ );
    r.fill(0.f);
    if( unary_ ) {
        MatrixXf unary = unary_->get();

        for( int i=0; i<N_; i++ )
            if ( 0 <= l[i] && l[i] < M_ )
                r[i] = unary( l[i], i );
    }
    return r;
}
VectorXf DenseCRF::pairwiseEnergy(const VectorXs & l, int term) const{
    assert( l.rows() == N_ );
    VectorXf r( N_ );
    r.fill(0.f);

    if( term == -1 ) {
        for( unsigned int i=0; i<pairwise_.size(); i++ )
            r += pairwiseEnergy( l, i );
        return r;
    }
    // This adds a negative term to the pairwise energy
    // and divide by two, I don't really know why.
    MatrixXf Q( M_, N_ );
    // Build the current belief [binary assignment]
    for( int i=0; i<N_; i++ )
        for( int j=0; j<M_; j++ )
            Q(j,i) = (l[i] == j);
    pairwise_[ term ]->apply( Q, Q );
    for( int i=0; i<N_; i++ )
        if ( 0 <= l[i] && l[i] < M_ )
            r[i] =-0.5*Q(l[i],i );
        else
            r[i] = 0;
    return r;
}

// true pairwise
VectorXf DenseCRF::pairwise_energy_true(const VectorXs & l, int term) const{
    assert( l.rows() == N_ );
    VectorXf r( N_ );
    r.fill(0);

    if( term == -1 ) {
        for( unsigned int i=0; i<no_norm_pairwise_.size(); i++ )
            r += pairwise_energy_true( l, i );
        return r;
    }

    MatrixXf Q( M_, N_ );
    // binary assignment (1 if l[i]!= j and 0 otherwise)
    for( int i=0; i<N_; i++ )
        for( int j=0; j<M_; j++ )
            Q(j,i) = (l[i] == j) ? 0 : 1;	// new
//            Q(j,i) = (l[i] == j) ? 1 : 0;	// old
    no_norm_pairwise_[ term ]->apply( Q, Q );
    for( int i=0; i<N_; i++ )
        if ( 0 <= l[i] && l[i] < M_ )
            r[i] = -Q(l[i],i );	// neg to cancel the neg introduced in compatiblity_.apply()
//            r[i] = Q(l[i],i );	// old
        else
            r[i] = 0;
    return r;
}

MatrixXf DenseCRF::startInference() const{
    MatrixXf Q( M_, N_ );
    Q.fill(0);

    // Initialize using the unary energies
    if( unary_ )
        expAndNormalize( Q, -unary_->get() );
    return Q;
}
void DenseCRF::stepInference( MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2 ) const{
    tmp1.resize( Q.rows(), Q.cols() );
    tmp1.fill(0);
    if( unary_ )
        tmp1 -= unary_->get();

    // Add up all pairwise potentials
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp2, Q );
        tmp1 -= tmp2;
    }

    // Exponentiate and normalize
    expAndNormalize( Q, tmp1 );
}
VectorXs DenseCRF::currentMap( const MatrixXf & Q ) const{
    VectorXs r(Q.cols());
    // Find the map
    for( int i=0; i<N_; i++ ){
        int m;
        Q.col(i).maxCoeff( &m );
        r[i] = m;
    }
    return r;
}

double DenseCRF::klDivergence( const MatrixXf & Q ) const {
    // Compute the KL-divergence of a set of marginals
    double kl = 0;
    // Add the entropy term
    for( int i=0; i<Q.cols(); i++ )
        for( int l=0; l<Q.rows(); l++ )
            kl += Q(l,i)*log(std::max( Q(l,i), 1e-20f) );
    // Add the unary term
    if( unary_ ) {
        MatrixXf unary = unary_->get();
        for( int i=0; i<Q.cols(); i++ )
            for( int l=0; l<Q.rows(); l++ )
                kl += unary(l,i)*Q(l,i);
    }

    // Add all pairwise terms
    MatrixXf tmp;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q );
        kl += (Q.array()*tmp.array()).sum();
    }
    return kl;
}

double DenseCRF::klDivergence( const MatrixXf & Q, const MatrixXf & P ) const {
    double kl = 0;
    assert(valid_probability_debug(Q));
    assert(valid_probability_debug(P));
    for (int i = 0; i < Q.cols(); ++i) {
        for (int j = 0; j < Q.rows(); ++j) {
            kl += std::max(Q(j, i), 1e-20f) * log(std::max(Q(j, i), 1e-20f) / std::max(P(j, i), 1e-20f));
        }
    }
    return kl;
}

double DenseCRF::compute_LR_QP_value(const MatrixXf & Q, const MatrixXf & diag_dom) const{
    double energy = 0;
    // Add the unary term
    MatrixXf unary = unary_->get();
    MatrixP dot_tmp;

    energy += dotProduct(unary, Q, dot_tmp);
    energy -= dotProduct(diag_dom, Q, dot_tmp);

    // Add all pairwise terms
    MatrixXf tmp;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q );
        energy += dotProduct(Q, tmp, dot_tmp);
    }
    energy += dotProduct(Q, diag_dom.cwiseProduct(Q), dot_tmp);
    return energy;
}

double DenseCRF::compute_energy(const MatrixXf & Q) const {
    double energy = 0;
    MatrixP dot_tmp;
    // Add the unary term
    if( unary_ ) {
        MatrixXf unary = unary_->get();
        energy += dotProduct(unary, Q, dot_tmp);
    }
    // Add all pairwise terms
    MatrixXf tmp;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q );
        energy += dotProduct(Q, tmp, dot_tmp);
    }
    return energy;
}

double DenseCRF::compute_energy_true(const MatrixXf & Q) const {
    double energy = 0;
    MatrixP dot_tmp;
	// Add the unary term
    if( unary_ ) {
        MatrixXf unary = unary_->get();
        energy += dotProduct(unary, Q, dot_tmp);
    }
    // Add all pairwise terms
    MatrixXf tmp, tmp2;
    for( unsigned int k=0; k<no_norm_pairwise_.size(); k++ ) {
#if BRUTE_FORCE
        no_norm_pairwise_[k]->apply_bf( tmp, Q );
#else
        no_norm_pairwise_[k]->apply( tmp, Q );
#endif
		energy += dotProduct(Q, tmp, dot_tmp);	// do not cancel the neg intoduced in apply
		// constant term
		tmp = -tmp;	// cancel the neg introdcued in apply
		tmp.transposeInPlace();
		tmp2 = Q*tmp;	
		double const_energy = tmp2.sum();
		energy += const_energy;
    }

    return energy;
}

double DenseCRF::compute_energy_LP(const MatrixXf & Q) const {
    assert(pairwise_.size() == no_norm_pairwise_.size());
    double energy = 0;
    MatrixP dot_tmp;
    MatrixXi ind(M_, N_);
    // Add the unary term
    if( unary_ ) {
        MatrixXf unary = unary_->get();
        energy += dotProduct(unary, Q, dot_tmp);
    }
	//std::cout << "\nph-unary-energy: " << energy;
    // Add all pairwise terms
    MatrixXf tmp(Q.rows(), Q.cols());
#if BRUTE_FORCE
    MatrixXf tmp2(Q.rows(), Q.cols());
    sortRows(Q, ind);
#else
    MatrixXf rescaled_Q(Q.rows(), Q.cols());
    rescale(rescaled_Q, Q);
#endif
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        // Add the upper minus the lower
#if BRUTE_FORCE
        no_norm_pairwise_[k]->apply_upper_minus_lower_bf(tmp2, ind);
        // need to sort before dot-product
        for(int i=0; i<tmp2.cols(); ++i) {
        	for(int j=0; j<tmp2.rows(); ++j) {
            	tmp(j, ind(j, i)) = tmp2(j, i);
        	}
        }
#else             
        // new-PH
        pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_Q);
#endif        
        energy -= dotProduct(Q, tmp, dot_tmp);
		//std::cout << "\nph-pairwise[" << k << "]-energy: " << -dotProduct(Q, tmp, dot_tmp);
    }

    return energy;
}

double DenseCRF::gradient( int n_iterations, const ObjectiveFunction & objective, VectorXf * unary_grad, VectorXf * lbl_cmp_grad, VectorXf * kernel_grad) const {
    // Gradient computations
    // Run inference
    std::vector< MatrixXf > Q(n_iterations+1);
    MatrixXf tmp1, unary( M_, N_ ), tmp2;
    unary.fill(0);
    if( unary_ )
        unary = unary_->get();
    expAndNormalize( Q[0], -unary );
    for( int it=0; it<n_iterations; it++ ) {
        tmp1 = -unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp2, Q[it] );
            tmp1 -= tmp2;
        }
        expAndNormalize( Q[it+1], tmp1 );
    }

    // Compute the objective value
    MatrixXf b( M_, N_ );
    double r = objective.evaluate( b, Q[n_iterations] );
    sumAndNormalize( b, b, Q[n_iterations] );

    // Compute the gradient
    if(unary_grad && unary_)
        *unary_grad = unary_->gradient( b );
    if( lbl_cmp_grad )
        *lbl_cmp_grad = 0*labelCompatibilityParameters();
    if( kernel_grad )
        *kernel_grad = 0*kernelParameters();

    for( int it=n_iterations-1; it>=0; it-- ) {
        // Do the inverse message passing
        tmp1.fill(0);
        int ip = 0, ik = 0;
        // Add up all pairwise potentials
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            // Compute the pairwise gradient expression
            if( lbl_cmp_grad ) {
                VectorXf pg = pairwise_[k]->gradient( b, Q[it] );
                lbl_cmp_grad->segment( ip, pg.rows() ) += pg;
                ip += pg.rows();
            }
            // Compute the kernel gradient expression
            if( kernel_grad ) {
                VectorXf pg = pairwise_[k]->kernelGradient( b, Q[it] );
                kernel_grad->segment( ik, pg.rows() ) += pg;
                ik += pg.rows();
            }
            // Compute the new b
            pairwise_[k]->applyTranspose( tmp2, b );
            tmp1 += tmp2;
        }
        sumAndNormalize( b, tmp1.array()*Q[it].array(), Q[it] );

        // Add the gradient
        if(unary_grad && unary_)
            *unary_grad += unary_->gradient( b );
    }
    return r;
}
VectorXf DenseCRF::unaryParameters() const {
    if( unary_ )
        return unary_->parameters();
    return VectorXf();
}
void DenseCRF::setUnaryParameters( const VectorXf & v ) {
    if( unary_ )
        unary_->setParameters( v );
}
VectorXf DenseCRF::labelCompatibilityParameters() const {
    std::vector< VectorXf > terms;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            terms.push_back( pairwise_[k]->parameters() );
        int np=0;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            np += terms[k].rows();
        VectorXf r( np );
        for( unsigned int k=0,i=0; k<pairwise_.size(); k++ ) {
            r.segment( i, terms[k].rows() ) = terms[k];
            i += terms[k].rows();
        }
        return r;
    }
    void DenseCRF::setLabelCompatibilityParameters( const VectorXf & v ) {
        std::vector< int > n;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            n.push_back( pairwise_[k]->parameters().rows() );
        int np=0;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            np += n[k];

        for( unsigned int k=0,i=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->setParameters( v.segment( i, n[k] ) );
            i += n[k];
        }
    }
    VectorXf DenseCRF::kernelParameters() const {
        std::vector< VectorXf > terms;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            terms.push_back( pairwise_[k]->kernelParameters() );
        int np=0;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            np += terms[k].rows();
        VectorXf r( np );
        for( unsigned int k=0,i=0; k<pairwise_.size(); k++ ) {
            r.segment( i, terms[k].rows() ) = terms[k];
            i += terms[k].rows();
        }
        return r;
    }
    void DenseCRF::setKernelParameters( const VectorXf & v ) {
        std::vector< int > n;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            n.push_back( pairwise_[k]->kernelParameters().rows() );
        int np=0;
        for( unsigned int k=0; k<pairwise_.size(); k++ )
            np += n[k];

        for( unsigned int k=0,i=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->setKernelParameters( v.segment( i, n[k] ) );
            i += n[k];
        }
    }

    void DenseCRF::compute_kl_divergence(){
        compute_kl = true;
    }
