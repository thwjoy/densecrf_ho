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
#include "msImageProcessor.h"
#include "libppm.h"
#include "util.h"
#include "pairwise.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>



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
}
DenseCRF2D::DenseCRF2D(int W, int H, int M) : DenseCRF(W*H,M), W_(W), H_(H) {
}
DenseCRF2D::~DenseCRF2D() {
}
/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::addPairwiseEnergy (const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type, NormalizationType normalization_type) {
    assert( features.cols() == N_ );
    addPairwiseEnergy( new PairwisePotential( features, function, kernel_type, normalization_type ) );
}
void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential ){
    pairwise_.push_back( potential );
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

void DenseCRF2D::addSuperPixel(unsigned char * img, int spatial_radius, int range_radius, int min_region_count, SpeedUpLevel) {
    //addSuperPixel is a member function that applies the mean-shift algorithm to the image and then initialises the protected member varaiable super_pixel_classifer.
    unsigned char * segment_image = new unsigned char[W_ * H_ * 3];
    int ** regions_out;
    float ** modes_out;
    int ** MPC_out;
    int region;

    msImageProcessor m_process;
    m_process.DefineImage(img , COLOR , H_ , W_);
    m_process.Segment(spatial_radius,range_radius,min_region_count,NO_SPEEDUP);
    m_process.GetResults(segment_image);
    R_ = m_process.GetRegions(regions_out,modes_out,MPC_out);
    super_pixel_classifier_.resize(R_,W_ * H_);
    super_pixel_classifier_.setZero();
    for (int i = 0; i < super_pixel_classifier_.cols(); i++) {
        region = regions_out[0][i];
        super_pixel_classifier_(region,i) = 1;
    }
    
    writePPMImage("./ouput.ppm",segment_image, H_, W_, 3, "");
    return;
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

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::pair<T,T> p)  
{  
        std::cout << "'" << p.first << "," << p.second << "'\r\n";
        return os;  
} 

std::ostream& operator<<(std::ostream& os, const std::vector<perf_measure>& perf)  
{  
        for (const auto & i : perf) {
            std::cout << i << "\r\n";
        } 
        return os;  
}



std::vector<perf_measure> DenseCRF::tracing_inference(MatrixXf & init, double time_limit) const {
    MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2, old_Q(M_, N_);
    float old_kl, kl;

    clock_t start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;

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
        perf_energy = assignment_energy(currentMap(Q));
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

    double num, denom;
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

        num =  2 * dotProduct(Q, psisx, temp_dot) + dotProduct(unary, sx, temp_dot);
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        
        denom = dotProduct(sx, psisx, temp_dot);

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
    perf_energy = assignment_energy(currentMap(Q));
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
        perf_energy = assignment_energy(currentMap(Q));
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

MatrixXf DenseCRF::qp_inference_non_convex(const MatrixXf & init) const {
    //Here we compute the Frank Wolfe algorithm on a non-convex energy function min phi' * y + y' * psi * y eq(8)
    //gradient = phi + 2 * psi * y 
    MatrixXf Q(M_, N_), unary(M_, N_),  tmp(M_,N_), grad(M_, N_),
        cond_grad(M_,N_), sx(M_,N_), psisx(M_, N_);
    MatrixP temp_dot(M_,N_);

    double optimal_step_size = 0;
    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    Q = init;

    // Compute the value of the energy
    double old_energy = std::numeric_limits<double>::max();
    double energy;

    //this computes the  gradient function phi + 2 * psi * y
    grad = unary;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q);
        grad += 2 *tmp;
    }
    
    int i = 0;
    energy = compute_LR_QP_value(Q, MatrixXf::Zero(M_,N_)); //normally takes the diagonally dominant component which in this case does not exist
    while( (old_energy - energy) > 1){
        old_energy = energy;
        i++;

        //solve the conditional gradient
        descent_direction(cond_grad, grad);

        //Solve for the best step size. The best step size is:
        //         1        phi' * (s - y) + 2 * y' * psi * (s - y)
        //    a = --- x ---------------------------------------------
        //         2              (s - y)' * psi * (s - y)
        //clearly the expensive term are  "psi * (s - y)", hence we create two new variables sx  = (s - y) amd psisx = psi * (s - y)

        sx = cond_grad - Q;
        psisx.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx);
            psisx += tmp;
        }
        

        double num =  2 * dotProduct(Q, psisx, temp_dot) + dotProduct(unary, sx, temp_dot);
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx, psisx, temp_dot);
        // Denom should be negative, as our energy function is now concave.
        
        optimal_step_size = - num / (2 * denom);
           
        if (denom == 0 || num == 0) {
            // This means that the conditional gradient is the same
            // than the current step and we have converged.
            optimal_step_size = 0;
            // Compute the gradient at the new estimates.
            grad += 2 * optimal_step_size * psisx;
            //energy = compute_LR_QP_value(Q, diag_dom);
            //alt_energy = dotProduct(Q, unary, temp_dot) + 0.5*dotProduct(Q, grad - unary, temp_dot);
            energy = 0.5 * dotProduct(Q, grad + unary, temp_dot);
            break;
        }

        if (optimal_step_size > 1) {
            // Respect the bounds.
            optimal_step_size = 0;
        }
        if (optimal_step_size < 0) {
            // Stay between the current step and the optimal.
            // Theoretically shouldn't happen but we accumulate
            // floating point errors when we compute the polynomial
            // coefficients.
            optimal_step_size = 1;
        }

        // Take a step
        Q += optimal_step_size * sx;
        if (not valid_probability(Q)) {
            std::cout << "Bad proba" << '\n';
        }
        // Compute the gradient at the new estimates.
        grad += 2 * optimal_step_size * psisx;
        //energy = compute_LR_QP_value(Q, diag_dom);
        //alt_energy = dotProduct(Q, unary, temp_dot) + 0.5*dotProduct(Q, grad - unary, temp_dot);
        energy = 0.5 * dotProduct(Q, grad + unary, temp_dot);
        
    }

    std::cout << "---Found optimal soloution in: " << i << " iterations.\r\n";

    return Q;
}

MatrixXf DenseCRF::qp_inference_super_pixels_non_convex(const MatrixXf & init) const {
    /*Here we compute the Frank Wolfe, to find the optimum of the cost function contatining super pixels
     * The cost function is: phi'.y + y'.psi.y + K[z + (1 - z){1'.theta.(1 - y)}]
     * where theta reprosents a compatibility matrix for the super pixel regions
     * The gradient of the cost function is defined as:
     *      g_y = phi + 2psi.y - K[(1 - z).theta'.1]
     *      g_z = 1 - [1'.theta.(1 - y)]
     *
     * The conditional gradient is then givel by:
     *      c = argmin(g_y'.y,g_z'.z)
     *
     * We can the compute the optimal step size over y and z     
     *
     *
     */
    MatrixXf Q(M_, N_), unary(M_, N_),  tmp(M_,N_), grad_y(M_, N_), 
        cond_grad_y(M_,N_), grad_z(M_, R_), cond_grad_z(M_, R_),sx_z(M_,R_), sx_y(M_,N_), psisx(M_, N_), z_labels(M_,R_);
    MatrixP temp_dot(M_,N_);


    double K = 296;
    std::cout << "K:" << K << std::endl;
    grad_z.fill(0);
    cond_grad_z.fill(0);
    z_labels.fill(1); //indicates that in all super pixels there are pixels that do not take the same label

    double optimal_step_size = 0;
    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    Q = init;

    // Compute the value of the energy
    double old_energy = std::numeric_limits<double>::max();
    double energy;


    //initialise the gradient of z
    grad_z = K * (MatrixXf::Ones(M_,R_) + ((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()));
    descent_direction_z(cond_grad_z, grad_z);
    //this computes the  gradient function phi + 2 * psi * y
    grad_y = unary;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q);
        grad_y += 2 *tmp;
    }
    grad_y += K * (z_labels - MatrixXf::Ones(M_,R_)) * super_pixel_classifier_;

    
    int i = 0;
    energy = 0.5 * dotProduct(Q, grad_y + unary, temp_dot);
    energy += K * (z_labels.sum() + dotProduct((MatrixXf::Ones(M_,R_) - z_labels),((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()),temp_dot));
    while( (old_energy - energy) > 1){
        old_energy = energy;
        i++;

        //solve the conditional gradient
        descent_direction(cond_grad_y, grad_y);
        descent_direction_z(cond_grad_z, grad_z);

        sx_y = cond_grad_y - Q;
        sx_z = cond_grad_z - z_labels;

        psisx.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx_y);
            psisx += tmp;
        }
        

        double a = K * dotProduct(sx_z,((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()),temp_dot);

        double b = K * dotProduct((cond_grad_z - MatrixXf::Ones(M_,R_)) * super_pixel_classifier_,sx_y,temp_dot);

        double num = dotProduct(unary, sx_y, temp_dot) + 2 * dotProduct(Q, psisx, temp_dot) + a + b + K * sx_z.sum();
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx_y, psisx, temp_dot) + K * dotProduct(cond_grad_z - MatrixXf::Ones(M_,R_),((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()),temp_dot);
        // Denom should be negative, as our energy function is now concave.
        optimal_step_size = - num / (2 * denom);


        //check bounds for optimal step size
        if (denom == 0 || num == 0) {break;}
        if (optimal_step_size > 1) { optimal_step_size = 1;}
        if (optimal_step_size < 0) { optimal_step_size = 1;}


        // Take a step
        Q += optimal_step_size * sx_y;
        z_labels += optimal_step_size * sx_z;
        if (not valid_probability(Q)) {
            std::cout << "Bad proba" << '\n';
        }

        //compute the new gradient
        grad_y = unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            grad_y += 2 *tmp;
        }
        grad_y += K * (z_labels - MatrixXf::Ones(M_,R_)) * super_pixel_classifier_;
        grad_z = K * (MatrixXf::Ones(M_,R_) + ((Q - MatrixXf::Ones(M_, N_)) * super_pixel_classifier_.transpose()));


        //energy = compute_LR_QP_value(Q, diag_dom);
        //alt_energy = dotProduct(Q, unary, temp_dot) + 0.5*dotProduct(Q, grad - unary, temp_dot);
        energy = 0.5 * dotProduct(Q, grad_y + unary, temp_dot);
        energy += K * (z_labels.sum() + dotProduct((MatrixXf::Ones(M_,R_) - z_labels),((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()),temp_dot));
     
    }

    std::cout << "---Found optimal soloution in: " << i << " iterations.\r\n";

    return Q;
}

std::vector<perf_measure> DenseCRF::tracing_qp_inference_super_pixels_non_convex(MatrixXf & init) const {
    /*Here we compute the Frank Wolfe, to find the optimum of the cost function contatining super pixels
     * The cost function is: phi'.y + y'.psi.y + K[z + (1 - z){1'.theta.(1 - y)}]
     * where theta reprosents a compatibility matrix for the super pixel regions
     * The gradient of the cost function is defined as:
     *      g_y = phi + 2psi.y - K[(1 - z).theta'.1]
     *      g_z = 1 - [1'.theta.(1 - y)]
     *
     * The conditional gradient is then givel by:
     *      c = argmin(g_y'.y,g_z'.z)
     *
     * We can the compute the optimal step size over y and z     
     *
     *
     */
    MatrixXf Q(M_, N_), unary(M_, N_),  tmp(M_,N_), grad_y(M_, N_), 
        cond_grad_y(M_,N_), grad_z(M_, R_), cond_grad_z(M_, R_),sx_z(M_,R_), sx_y(M_,N_), psisx(M_, N_), z_labels(M_,R_);
    MatrixP temp_dot(M_,N_);

    clock_t start, end;
    double perf_energy, perf_timing;
    double total_time = 0;
    perf_measure latest_perf;
    std::vector<perf_measure> perfs;
 
    int K = 100;
    grad_z.fill(0);
    cond_grad_z.fill(0);
    z_labels.fill(1); //indicates that in all super pixels there are pixels that do not take the same label

    double optimal_step_size = 0;

   
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    Q = init;

    // Compute the value of the energy
    double old_energy;
    double energy = std::numeric_limits<double>::max();

    start = clock();

    {
        //initialise the gradient of z
        grad_z = K * (MatrixXf::Ones(M_,R_) + ((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()));
        descent_direction_z(cond_grad_z, grad_z);
        //this computes the  gradient function phi + 2 * psi * y
        grad_y = unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            grad_y += 2 *tmp;
        }
        grad_y += K * (cond_grad_z - MatrixXf::Ones(M_,R_)) * super_pixel_classifier_;

   
    } // This is some necessary setup for the QP inference so this need to be accounted.

    end = clock();
    perf_timing = (double(end-start)/CLOCKS_PER_SEC);
    perf_energy = assignment_energy(currentMap(Q));
    latest_perf = std::make_pair(perf_timing, perf_energy);
    perfs.push_back(latest_perf);
    total_time += perf_timing;

    int i = 0;
    do {
        old_energy = energy;
        i++;


        //solve the conditional gradient for y 
        descent_direction(cond_grad_y, grad_y);
        //the conditional gradient of z is a vector of 0's
        descent_direction_z(cond_grad_z, grad_z);

        //std::cout << z_labels;
  
        //sx  = (s - y) difference between conditional and actual
        //psisx = psi * (s - y)
        sx_y = cond_grad_y - Q;
        sx_z = cond_grad_z - z_labels;

        
        psisx.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx_y);
            psisx += tmp;
        }
         
        /* 
         * ######################### Optimal step size computation #########################
         * The optimal step size is given as:
         *
         * argmin_a f(y + a(c_y - y),z + a(c_z - z))
         *
         * argmin_a [phi' * (y + a(c_y - y)) + (y + a(c_y - y))' * psi * (y + a(c_y - y)) + K{(z + a(c_z - z))) + (1 - (z + a(c_z - z))) * theta * (1 - (y + a(c_y - y)))}]
         *
         * argmin_a     a*a[(c_y - y)' * psi * (c_y - y) + K{(c_z - z) * theta * (c_y - y)}] 
         * + a[phi' * (c_y - y) + 2y' * psi * (c_y - y) * K{(c_z - z) - (c_z - z) * theta * 1_y - 1_z * 1_y * theta * (c_y - y)}]
         * + [phi' * y + y' * psi * y + K{z + 1_z * 1_t * theta * 1_y}]
         *
         *  This is effectively a quadratic equation with a being the varaible, the minimum of lx^2 + mx + n is givend by x = 0.5 * m / l
         *  Thus the optimal step size of our problem is given by:
         *
         *        1     phi' * (c_y - y) + K{(c_z - z) - (c_z - z)* 1_t * theta * 1_y - 1_z * 1_y * theta * (c_y - y)}]
         *   a = --- x ---------------------------------------------------------------------------------------------------
         *        2              (c_y - y)' * psi * (c_y - y) + K{(c_z - z) * 1 * theta * (c_y - y)}
         *
         *
         * #################################################################################
         */

        double a = K * dotProduct(sx_z,((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()),temp_dot);
        double b = K * dotProduct((cond_grad_z - MatrixXf::Ones(M_,R_)) * super_pixel_classifier_,sx_y,temp_dot);
           
        double num = dotProduct(unary, sx_y, temp_dot) + 2 * dotProduct(Q, psisx, temp_dot) + a + b + K * sx_z.sum();
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx_y, psisx, temp_dot) + K * dotProduct(cond_grad_z - MatrixXf::Ones(M_,R_),((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()),temp_dot);
        // Denom should be negative, as our energy function is now concave.
        
        optimal_step_size = - num / (2 * denom);
        //optimal_step_size = 0.5;

        std::cout << "optimal_step_size: " << optimal_step_size << std::endl;
           
        if (denom == 0 || num == 0) {
            // This means that the conditional gradient is the same
            // than the current step and we have converged.
            break;
        }

        if (optimal_step_size > 1) { optimal_step_size = 1;}
        if (optimal_step_size < 0) { optimal_step_size = 1;}

        
        // we have to do this before we take a step as it means we only compute the expensive psi * y term once.   
        grad_y -= K * (z_labels - MatrixXf::Ones(M_,R_)) * super_pixel_classifier_;
        // Take a step
        Q += optimal_step_size * sx_y;
        z_labels += optimal_step_size * sx_z;

        if (not valid_probability(Q)) {std::cout << "Bad proba" << '\n';}



        //compute the gradient updates
        grad_y += 0.5 * optimal_step_size * psisx;
        grad_y += K * (z_labels - MatrixXf::Ones(M_,R_)) * super_pixel_classifier_;

        grad_z = K * (MatrixXf::Ones(M_,R_) + ((Q - MatrixXf::Ones(M_, N_)) * super_pixel_classifier_.transpose()));
    
        
       //compute the energy
        energy = 0.5 * dotProduct(Q, grad_y + unary, temp_dot); 
        energy += K * (z_labels.sum() + dotProduct((MatrixXf::Ones(M_,R_) - z_labels),((Q - MatrixXf::Ones(M_,N_)) * super_pixel_classifier_.transpose()),temp_dot));
        
        // performance measurement
        end = clock();
        perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        perf_energy = assignment_energy(currentMap(Q));
        latest_perf = std::make_pair(perf_timing, perf_energy);
        perfs.push_back(latest_perf);
        total_time += perf_timing;
    } while((old_energy - energy) > 1 && i < 50);
    
    std::cout << "---Found optimal soloution in: " << i << " iterations.\r\n";
    init = Q;
    std::cout << perfs;
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
        perf_energy = assignment_energy(currentMap(Q));
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
    perf_energy = assignment_energy(currentMap(Q));
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
        perf_energy = assignment_energy(currentMap(Q));
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

VectorXs get_original_label(VectorXs const & restricted_labels, std::vector<int> const & indices) {
    VectorXs extended_labels(restricted_labels.rows());
    for(int i=0; i<restricted_labels.rows(); i++) {
        extended_labels[i] = indices[restricted_labels[i]];
    }
    return extended_labels;
}

MatrixXf DenseCRF::lp_inference(MatrixXf & init, bool use_cond_grad) const {
    // Random init to prevent too many elements to be 0
    // init.setRandom();
    // MatrixXf uns(init.rows(), init.cols());
    // uns.fill(1);
    // init = init + uns;
    // renormalize(init);

    // Restrict number of labels in the computation
    std::vector<int> indices;
    renormalize(init);
    get_limited_indices(init, indices);
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
                /*start = clock();*/
                no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp, ind);
                tmp2.fill(0);
                for(i=0; i<tmp.cols(); ++i) {
                    for(j=0; j<tmp.rows(); ++j) {
                        tmp2(j, ind(j, i)) = tmp(j, i);
                    }
                }
                /*end = clock();
                float perf_timing = (double(end-start)/CLOCKS_PER_SEC);
                printf("DC: It: %d | id: %d | time: %f\n", it, k, perf_timing);*/

                // With the new discretized split computations
                // start = clock();
                //pairwise_[k]->apply_upper_minus_lower_ord(tmp, Q);
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
    } while(it<100);
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

    ones.fill(1);

    int i,j;
    int nb_pairwise = pairwise_.size();

    // Create copies of the original pairwise since we don't want normalization
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

    best_Q = Q;

    // Compute the value of the energy
    assert(valid_probability(Q));
    double energy = 0, best_energy = std::numeric_limits<double>::max();
    double int_energy = assignment_energy(currentMap(Q));
    // printf("Initial int energy in the LP: %f\n", int_energy);


    clock_t start, end;

    int it=0;
    do {
        ++it;

        // Compute the current energy and gradient
        // Unary
        energy = dotProduct(unary, Q, dot_tmp);
        grad = unary;

        // Pairwise
        sortRows(Q, ind);
        for( unsigned int k=0; k<nb_pairwise; k++ ) {
            // Add upper minus lower
            /*no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp, ind);
            tmp2.fill(0);
            for(i=0; i<tmp.cols(); ++i) {
                for(j=0; j<tmp.rows(); ++j) {
                    tmp2(j, ind(j, i)) = tmp(j, i);
                }
            }*/
            pairwise_[k]->apply_upper_minus_lower_ord(tmp2, Q);
            energy -= dotProduct(Q, tmp2, dot_tmp);
            grad -= tmp2;
        }

        // Print previous iteration energy
        printf("%4d: %10.3f / %10.3f\n", it-1, energy, int_energy);
        if( energy < best_energy) {
            best_Q = Q;
            best_energy = energy;
        }
        // printf("%5d: %f\n", it-1, energy);

        // Sub-gradient descent step
        // std::cout<<grad.block(0,0,5,5)<<std::endl<<std::endl;
        // std::cout<<Q.block(0,0,5,5)<<std::endl<<std::endl;
        Q -= grad/(it+1e5);

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

        int_energy = assignment_energy(currentMap(Q));
        renormalize(Q);
        assert(valid_probability_debug(Q));
    } while(it<100);
    std::cout <<"final projected energy: " << int_energy << "\n";

    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        delete no_norm_pairwise[k];
    }
    free(no_norm_pairwise);

    return best_Q;
}


std::vector<perf_measure> DenseCRF::tracing_lp_inference(MatrixXf & init, bool use_cond_grad, double time_limit) const {
    // Restrict number of labels in the computation
    std::vector<int> indices;
    get_limited_indices(init, indices);
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
    double best_int_energy = assignment_energy(get_original_label(currentMap(Q), indices));

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

        end = clock();
        perf_timing = (double(end-start)/CLOCKS_PER_SEC);
        perf_energy = assignment_energy(currentMap(get_extended_matrix(best_Q, indices, M_)));
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

// only calculate pairwise energies -- assumes single kernel
void DenseCRF::compare_energies(const MatrixXf & Q, double & ph_energy, double & bf_energy, 
		bool qp, bool ph_old) const {
	if (pairwise_.size() != 1) {
		std::cout << "pairwise-size: " << pairwise_.size() << " (should be 1)" << std::endl;
		exit(1);
	}
	if (!valid_probability(Q)) {
		std::cout << "Q is not a valid probability!" << std::endl;
		exit(1);
	}
	
	// for bruteforce computation
	// Create copies of the original pairwise since we don't want normalization
    PairwisePotential** no_norm_pairwise;
    no_norm_pairwise = (PairwisePotential**) malloc(pairwise_.size()*sizeof(PairwisePotential*));
	const int k = 0;	//fixed
    no_norm_pairwise[k] = new PairwisePotential(
        pairwise_[k]->features(),
        new PottsCompatibility(pairwise_[k]->parameters()(0)),
        pairwise_[k]->ktype(),
        NO_NORMALIZATION
        );
	
    MatrixXf tmp(M_, N_), tmp2(M_, N_);
	MatrixXi ind(M_, N_);
	MatrixP dot_tmp;
	double energy = 0;

	if (qp) {
		// ph-energy
		energy = 0;
        pairwise_[k]->apply( tmp, Q );
    	energy += dotProduct(Q, tmp, dot_tmp);	// do not cancel the neg intoduced in apply
    	// constant term
    	tmp = -tmp;	// cancel the neg introdcued in apply
    	tmp.transposeInPlace();
    	tmp2 = Q*tmp;	
    	double const_energy = tmp2.sum();
    	energy += const_energy;
		ph_energy = energy;

		//bf-energy
		energy = 0;
		no_norm_pairwise[k]->apply_bf( tmp, Q );
		energy += dotProduct(Q, tmp, dot_tmp);	// do not cancel the neg intoduced in apply
		// constant term
		tmp = -tmp;	// cancel the neg introdcued in apply
		tmp.transposeInPlace();
		tmp2 = Q*tmp;	
		const_energy = tmp2.sum();
		energy += const_energy;
		bf_energy = energy;

	} else {
		// ph-energy
		energy = 0;
		// old-ph
		if (ph_old) {
    		sortRows(Q, ind);
            no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp2, ind);
    		// need to sort before dot-product
    		for(int i=0; i<tmp2.cols(); ++i) {
            	for(int j=0; j<tmp2.rows(); ++j) {
                	tmp(j, ind(j, i)) = tmp2(j, i);
            	}
            }
		} else {
        	// Add the upper minus the lower
	        pairwise_[k]->apply_upper_minus_lower_ord(tmp, Q);
		}
		//
        energy -= dotProduct(Q, tmp, dot_tmp);
		ph_energy = energy;

		// bf-energy
		energy = 0;
		sortRows(Q, ind);
        no_norm_pairwise[k]->apply_upper_minus_lower_bf(tmp2, ind);
		// need to sort before dot-product
		for(int i=0; i<tmp2.cols(); ++i) {
        	for(int j=0; j<tmp2.rows(); ++j) {
            	tmp(j, ind(j, i)) = tmp2(j, i);
        	}
        }
        energy -= dotProduct(Q, tmp, dot_tmp);
		bf_energy = energy;
	}
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

MatrixXf DenseCRF::interval_rounding(const MatrixXf &estimates) const {
    int nb_random_rounding = 10;

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

double DenseCRF::compute_energy_LP(const MatrixXf & Q, PairwisePotential** no_norm_pairwise, int nb_pairwise) const {
    double energy = 0;
    MatrixP dot_tmp;
    MatrixXi ind(M_, N_);
    // Add the unary term
    if( unary_ ) {
        MatrixXf unary = unary_->get();
        energy += dotProduct(unary, Q, dot_tmp);
    }
    // Add all pairwise terms
    sortRows(Q, ind);
    MatrixXf tmp(Q.rows(), Q.cols());
    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        // Add the upper minus the lower
        no_norm_pairwise[k]->apply_upper_minus_lower_dc(tmp, ind);
        assert(tmp.maxCoeff()<1e-3);
        energy -= dotProduct(Q, tmp, dot_tmp);
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
