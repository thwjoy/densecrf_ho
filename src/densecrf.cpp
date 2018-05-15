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
#include "Eigen/LU"
#include "Eigen/Core"
#include "Eigen/src/Core/util/Constants.h"
#include "qp.hpp"
#include "permutohedral.h"
#include "msImageProcessor.h"
#include "libppm.h"
#include "util.h"
#include "pairwise.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <fstream>
#include <set>

#include <limits>

#define DIFF_ENG 100
/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////


float DenseCRF::multiplySuperPixels(const MatrixXf & p1, const MatrixXf & p2) const {
    float res = 0;
    assert(p1.cols() == R_);
    assert(p1.rows() == M_);
    assert(p2.cols() == N_);
    assert(p2.rows() == M_);
    for(int sp_reg = 0; sp_reg < R_; sp_reg++) { //for each super pixel
        for(int pixel = 0; pixel < super_pixel_container_[sp_reg].size(); pixel++) { //sum up the values where the pixel is in the sp
            for(int lab = 0; lab < M_; lab++) {
                res += p1(lab, sp_reg) * p2(lab, pixel);
            }
        }
    }
    return res;
}

MatrixXf DenseCRF::multiplySuperPixels(const MatrixXf & p) const {
    int dim;
    if (p.cols() == R_) dim = N_;
    else dim = R_;    //if we pass in a z vector the size needs to be that of the y vector
    assert(p.rows() == M_);
    MatrixXf res(M_, dim);
    res.fill(0);
    if (dim == R_) { //we want to sum up over the pixel values
        for(int sp_reg = 0; sp_reg < R_; sp_reg++) {
            for(int lab =0; lab < M_; lab++) {
                for(int pixel = 0; pixel < super_pixel_container_[sp_reg].size(); pixel++) {
                    res(lab,sp_reg) += p(lab,pixel);
                }
            }
        }

    } else { //we want to sum up over the
        for(int sp_reg = 0; sp_reg < R_; sp_reg++) {
            for(int lab =0; lab < M_; lab++) {
                for(int pixel = 0; pixel < super_pixel_container_[sp_reg].size(); pixel++) {
                    res(lab,pixel) += p(lab,sp_reg);
                }
            }
        }
    }

    return res;
    
} 

float DenseCRF::multiplyDecompSuperPixels(const MatrixXf & p1, const MatrixXf & p2, int sp_reg) const {
    float res = 0;
    for(int pixel = 0; pixel < super_pixel_container_[sp_reg].size(); pixel++) { //sum up the values where the pixel is in the sp
        for(int lab = 0; lab < M_; lab++) {
            res += p1(lab, sp_reg) * p2(lab, pixel);
        }
    }
    return res;
}

MatrixXf DenseCRF::multiplyDecompSuperPixels(const MatrixXf & p, int sp_reg) const {
    //if we just get a single columed vector then we are just summin up over labels and hence the output will just be a scalar
    int dim = 1;
    if (p.cols() == 1) dim = super_pixel_container_[sp_reg].size();
    assert(p.rows() == M_);
    MatrixXf res(M_, dim);
    res.fill(0);
    if (dim == 1) { //we want to sum up over the pixel values resulting in a single vector of size M_ * 1
        for(int lab =0; lab < M_; lab++) {
            for(int pixel = 0; pixel < super_pixel_container_[sp_reg].size(); pixel++) {
                res(lab,sp_reg) += p(lab,pixel);
            }
        }
        
    } else { //we want to sum up over the label values 
        for(int lab =0; lab < M_; lab++) {
            for(int pixel = 0; pixel < super_pixel_container_[sp_reg].size(); pixel++) {
                res(lab,pixel) += p(lab,sp_reg);
            }
        }      
    }
    return res;
}


DenseCRF::DenseCRF(int N, int M) : N_(N), M_(M), unary_(0), R_(0) {
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
    addPairwiseEnergy(feature, function, kernel_type, normalization_type);
}


void DenseCRF2D::addSuperPixel(unsigned char * img, int spatial_radius, int range_radius, int min_region_count, float constant, float normaliser) {

    //addSuperPixel is a member function that applies the mean-shift algorithm to the image and then initialises the protected member varaiable super_pixel_classifer.
    unsigned char * segment_image = new unsigned char[W_ * H_ * 3];
    std::vector<int> regions_out;
    std::vector<std::vector<double>> super_pixel_container;
    VectorXf count_regions; 
    VectorXf mean_of_superpixels;
    VectorXf sd_of_superpixels; 
    int region;
    Matrix<float, Dynamic, Dynamic> super_pixel_classifier;

    //get the mean shift info
    msImageProcessor m_process;
    m_process.DefineImage(img , COLOR , H_ , W_);
    m_process.Segment(spatial_radius, range_radius, min_region_count, NO_SPEEDUP);
    m_process.GetResults(segment_image);
    int reg = m_process.GetRegions(regions_out);

    super_pixel_container.resize(reg);
    count_regions.resize(reg);
    mean_of_superpixels.resize(reg);
    sd_of_superpixels.resize(reg);
    count_regions.fill(0);
    mean_of_superpixels.fill(0);
    sd_of_superpixels.fill(0);
    for (int i = 0; i < H_ * W_; i++) {
        region = regions_out[i];
        super_pixel_container[region].push_back(i);
        mean_of_superpixels(region) += 0.2989 * (double) img[i];
        mean_of_superpixels(region) += 0.5870 * (double) img[i + 1];
        mean_of_superpixels(region) += 0.1140 * (double) img[i + 2];
        count_regions(region) += 1;
    }

    mean_of_superpixels = mean_of_superpixels.cwiseQuotient(count_regions);

    for (int i = 0; i < regions_out.size(); i++) {
        region = regions_out[i];
        double grey_val = 0.2989 * (double) img[i] + 0.5870 * (double) img[i + 1] + 0.1140 * (double) img[i + 2];
        sd_of_superpixels(region) += (mean_of_superpixels(region) - grey_val) * (mean_of_superpixels(region) - grey_val);
        regions_out[i] += R_; //increment the current region so we don't get a mix up!
    }

    sd_of_superpixels = (sd_of_superpixels.cwiseQuotient(normaliser * count_regions));
    //only used for the LP
    std::vector<float> constants(reg, constant);
    constants_.insert(constants_.end(),constants.begin(),constants.end());

    R_ += reg;
    super_pixel_container_.insert(super_pixel_container_.end(),super_pixel_container.begin(), super_pixel_container.end());
    mean_of_superpixels_.conservativeResize(R_);
    mean_of_superpixels_.tail(reg) = mean_of_superpixels;
    exp_of_superpixels_.conservativeResize(R_);
    exp_of_superpixels_.tail(reg) = constant * exp(-1 * sd_of_superpixels.array());

    delete[] segment_image;
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


void expAndNormalize ( MatrixXf & out, const MatrixXf & in ) {
    out.resize( in.rows(), in.cols() );
    for( int i=0; i<out.cols(); i++ ){
        VectorXf b = in.col(i);
        b.array() -= b.maxCoeff();
        b = b.array().exp();
        out.col(i) = b / b.array().sum();
    }
}



MatrixXf DenseCRF::qp_inference_super_pixels_non_convex(const MatrixXf & init) {
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
     */

    MatrixXf Q(M_, N_), unary(M_, N_), diag_dom(M_,N_),  tmp(M_,N_), grad_y(M_, N_), 
        cond_grad_y(M_,N_), grad_z(M_, R_), cond_grad_z(M_, R_),sx_z(M_,R_), sx_y(M_,N_), psisx(M_, N_), z_labels(M_,R_);
    MatrixP temp_dot(M_,N_);
    MatrixXf constant = exp_of_superpixels_.replicate( 1, grad_z.rows() ).transpose();
    //Implement the exponentials in diagonal way

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
  
    //initialise the gradient of z
    grad_z = (MatrixXf::Ones(M_,R_) + multiplySuperPixels((Q - MatrixXf::Ones(M_,N_))));
    grad_z = grad_z.cwiseProduct(constant);
    descent_direction_z(cond_grad_z, grad_z);
    //this computes the  gradient function phi + 2 * psi * y
    grad_y = unary;
    for( unsigned int i=0; i<pairwise_.size(); i++ ) {
        pairwise_[i]->apply( tmp, Q);
        grad_y += 2 *tmp;
    }
    grad_y += multiplySuperPixels((z_labels - MatrixXf::Ones(M_,R_)).cwiseProduct(constant));
    
    // Compute the value of the energy
    double old_energy = std::numeric_limits<double>::max();
    double energy = compute_LR_QP_value(Q, 0 * MatrixXf::Ones(M_,N_));
    energy += (z_labels.cwiseProduct(constant).sum() + multiplySuperPixels((MatrixXf::Ones(M_,R_) - z_labels).cwiseProduct(constant),(Q - MatrixXf::Ones(M_,N_))));

    int i = 0;
    while((old_energy - energy) > DIFF_ENG){

        old_energy = energy;
        i++;

        //solve the conditional gradient
        descent_direction(cond_grad_y, grad_y);
        descent_direction_z(cond_grad_z, grad_z);

        sx_y = cond_grad_y - Q;
        sx_z = cond_grad_z - z_labels;

        psisx.fill(0);
        for( unsigned int i=0; i<pairwise_.size(); i++ ) {
            pairwise_[i]->apply(tmp, sx_y);
            psisx += tmp;
        }

        double a = multiplySuperPixels(sx_z.cwiseProduct(constant),(Q - MatrixXf::Ones(M_,N_)));

        double b = multiplySuperPixels((cond_grad_z - MatrixXf::Ones(M_,R_)).cwiseProduct(constant),sx_y);

        double num = dotProduct(unary, sx_y, temp_dot) + 2 * dotProduct(Q, psisx, temp_dot) + a + b + sx_z.sum();
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx_y, psisx, temp_dot) + multiplySuperPixels(sx_z.cwiseProduct(constant),sx_y);
        // Denom should be negative, as our energy function is now concave.
        optimal_step_size = - num / (2 * denom);

        //check bounds for optimal step size
        if (denom == 0 || num == 0) {break;}
        //std::cout << "Numerator: " << num << ", Denomonator: " << denom << ", Step size: " << optimal_step_size << std::endl;
        if (denom < 0) { optimal_step_size = 1;} //the function is concave and hence the optimal step size is 1

        // Take a step
        Q += optimal_step_size * sx_y;
        z_labels += optimal_step_size * sx_z;
        if (not valid_probability(Q)) {
            std::cout << "Bad proba" << '\n';
        }

        //compute the new gradient
        grad_y += 2 * optimal_step_size * psisx + optimal_step_size * multiplySuperPixels((z_labels - MatrixXf::Ones(M_,R_)).cwiseProduct(constant));
        grad_z = MatrixXf::Ones(M_,R_).cwiseProduct(constant) + (multiplySuperPixels(Q - MatrixXf::Ones(M_,N_))).cwiseProduct(constant);

        energy = dotProduct(Q, grad_y + unary, temp_dot);
        energy += (z_labels.cwiseProduct(constant).sum() + multiplySuperPixels((MatrixXf::Ones(M_,R_) - z_labels).cwiseProduct(constant),(Q - MatrixXf::Ones(M_,N_))));
    }

    //std::cout << "---Found optimal soloution in: " << i << " iterations.\r\n";
    z_labels_ = z_labels;

    return Q;
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

// LP inference with proximal algorithm
MatrixXf DenseCRF::lp_inference_prox_super_pixels(MatrixXf & init, LP_inf_params & params) const {
    MatrixXf best_Q(M_, N_), tmp(M_, N_), tmp2(M_, N_);
    MatrixP dot_tmp(M_, N_);
    MatrixXi ind(M_, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    MatrixXf unary = unary_->get();

    MatrixXf Q = init;
    renormalize(Q);
    //assert(valid_probability_debug(Q));
    best_Q = Q;

    // Compute the value of the energy
    double energy = 0, best_energy = std::numeric_limits<double>::max(), 
           best_int_energy = std::numeric_limits<double>::max();
    double int_energy = assignment_energy_higher_order(currentMap(Q));

    energy = compute_energy_LP_higher_order(Q);
    if (energy > int_energy) {  // choose the best initialization 
        Q = max_rounding(Q);
        energy = compute_energy_LP_higher_order(Q);
    }
    best_energy = energy;
    best_int_energy = int_energy;


    const int maxiter = params.prox_max_iter;
    const bool best_int = params.best_int;
    const bool accel_prox = params.accel_prox;
    const float prox_tol = params.prox_energy_tol;      // proximal energy tolerance

    // dual Frank-Wolfe variables
    const float dual_gap_tol = params.dual_gap_tol;     // dual gap tolerance
    const float lambda = params.prox_reg_const; // proximal-regularization constant
    const int fw_maxiter = params.fw_max_iter;
    float delta = 1;                        // FW step size
    double dual_gap = 0, dual_energy = 0;

    // dual variables
    MatrixXf alpha_tQ(M_, N_);  // A * alpha, (t - tilde not iteration)
    MatrixXf u_tQ(M_,N_);       // U * mew 
    MatrixXf sa_tQ(M_, N_);     // A * s, conditional gradient of FW == subgradient
    MatrixXf su_tQ(M_,N_);      // Conditional gradient for mew
    VectorXf beta(N_);          // unconstrained --> correct beta values (beta.row(i) == v_beta forall i)
    MatrixXf beta_mat(M_, N_);  // beta_mat.row(i) == beta forall i --> N_ * M_ elements 
    MatrixXf gamma(M_, N_);     // nonnegative
    
    MatrixXf cur_Q(M_, N_);     // current Q in prox step
    MatrixXf rescaled_Q(M_, N_);// infeasible Q rescaled to be within [0,1]


    // accelerated prox_lp
    MatrixXf prev_Q(M_, N_);    // Q resulted in prev prox step
    prev_Q.fill(0);
    float w_it = 1;             // momentum weight: eg. it/(it+3)

    // gamma-qp variables
    const float qp_delta = params.qp_const; // constant used in qp-gamma
    const float qp_tol = params.qp_tol;     // qp-gamma tolernace
    const int qp_maxiter = params.qp_max_iter;


    VectorXf v_gamma(M_), v_y(M_), v_pos_h(M_), v_neg_h(M_), v_step(M_), v_tmp(M_), v_tmp2(M_);
    MatrixXf Y(M_, N_), neg_H(M_, N_), pos_H(M_, N_);
    VectorXf qp_values(N_);

    //clock_t start, end;
    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;


    int it=0;
    int count = 0;


    do { //step iteration
        ++it;

        // initialization
        beta_mat.fill(0);
        beta.fill(0);
        gamma.fill(1);  // all zero is a fixed point of QP iteration!
        cur_Q = Q;
        
        
        int pit = 0;
        alpha_tQ.fill(0);   // all zero alpha_tQ is feasible --> alpha^1_{abi} = alpha^2_{abi} = K_{ab}/4
        u_tQ.fill(0);
        
        // proximal iteration this is the t iterating parameter in the paper
        do {    
            ++pit;
            // initialization
            sa_tQ.fill(0);
            su_tQ.fill(0);
/*#############################Optimise aver gamma##############################*/
            // QP-gamma -- \cite{NNQP solver Xiao and Chen 2014}
            // case-1: solve for gamma using qp solver! 
            // 1/2 * gamma^T * C * gamma - gamma^T * H
            // populate Y matrix (compute H in the paper)
            tmp = alpha_tQ + u_tQ - unary;
            for (int i = 0; i < N_; ++i) {
                // do it in linear time
                v_tmp2 = tmp.col(i);
                qp_gamma_multiplyC(v_y, v_tmp2, M_, lambda); //computes eq 35 in the paper for each pixel
                Y.col(i) = v_y; 
            }   

            Y += cur_Q;     // Y is the h in the paper given as h = Q(Aa - phi) - y^k (paper notation)
            //we have calculate Y as a +ve when it's actually -ve
            for (int i = 0; i < N_; ++i) {
                for (int j = 0; j < M_; ++j) {  
                    pos_H(j, i) = std::max(-Y(j, i), (float)0);     // pos_H 
                    neg_H(j, i) = std::max(Y(j, i), (float)0);      // neg_H
                }
            }
            // qp iterations, 
            int qpit = 0;
            qp_values.fill(0);
            float qp_value = std::numeric_limits<float>::max();

            
            double dt = 0;
            do {
                //solve for each pixel separately, this employes the algorithm from citeation 26
                for (int i = 0; i < N_; ++i) {
                    v_gamma = gamma.col(i); //col just accesses the lables of pixel i
                    v_pos_h = pos_H.col(i);
                    v_neg_h = neg_H.col(i);

                    //htime st = std::chrono::high_resolution_clock::now();
                    // do it linear time
                    qp_gamma_step(v_gamma, v_pos_h, v_neg_h, qp_delta, M_, lambda, v_step, v_tmp, v_tmp2);
                    //htime et = std::chrono::high_resolution_clock::now();
                    //dt += std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
                    gamma.col(i) = v_gamma;
                                   
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

            
            //std::cout << "Number of iterations: " << qpit << std::endl;
            
/*#############################Optimise aver beta###############################*/


            // case-2: update beta -- gradient of dual wrt beta equals to zero
            beta_mat = (alpha_tQ + u_tQ + gamma - unary);  // -B^T/l * (A * alpha + gamma - phi)
            // DON'T DO IT AT ONCE!! (RETURNS A SCALAR???)--> do it in two steps
            beta = -beta_mat.colwise().sum();   
            beta /= M_;
            // repeat beta in each row of beta_mat - - B * beta
            for (int j = 0; j < M_; ++j) {
                beta_mat.row(j) = beta;
            }

/*#############################Optimise aver alpha and mew##############################*/


            // case-3: dual conditional-gradient or primal-subgradient (do it as the final case)
            Q = lambda * (alpha_tQ + beta_mat + u_tQ + gamma - unary) + cur_Q; // Q may be infeasible --> but no problem
            try {
		rescale(rescaled_Q, Q);//truncate Q to be [0,1]
		}
		catch (std::runtime_error &e) {
			throw std::runtime_error(e.what());
		}
            // Compute the conditional gradient
            // subgradient lower minus upper
            // Pairwise
		
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
                // new PH implementation
                // rescaled_Q values in the range [0,1] --> but the same order as Q! --> subgradient of Q
                try {
		    pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_Q); 
                 sa_tQ += tmp;    // A * s is lower minus upper, keep neg introduced by compatibility->apply
               } catch (std::runtime_error & e) {
	      	    throw std::runtime_error(e.what());
		}	
            }
	    
            //Compute the conditional gradient for mew
            computeUCondGrad(su_tQ,Q);



            // find dual gap
            tmp = alpha_tQ + u_tQ - (sa_tQ + su_tQ);  
            dual_gap = dotProduct(tmp, Q, dot_tmp);

            // compute the optimal step size
            delta = (float)(dual_gap / (lambda * dotProduct(tmp, tmp, dot_tmp)));
            delta = std::min(std::max(delta, (float)0.0), (float)1.0);  // I may not need to truncate the step-size!!
//            if (delta == 0) {
//		std::cout << "step = 0" << std::endl;
//		//break;
//		}

            // update alpha_tQ
            alpha_tQ = alpha_tQ + delta * (sa_tQ - alpha_tQ);
            u_tQ = (u_tQ + delta * (su_tQ - u_tQ));

        } while(pit<fw_maxiter);

/*##############################################################################*/

  
        // project Q back to feasible region
        feasible_Q(tmp, ind, sum, K, Q);
        Q = tmp;
        renormalize(Q);
        //assert(valid_probability_debug(Q));

        double prev_int_energy = int_energy;
        int_energy = assignment_energy_higher_order(currentMap(Q));

        if (best_int) {
            if (abs(int_energy - prev_int_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                //std::cout << "\n##CONV: int_energy - prev_int_energy < " << prox_tol << " for last " << count
                //    << " iterations! terminating...\n";
                break;
            }
            if(int_energy < best_int_energy) {
                best_Q = Q;
                best_int_energy = int_energy;
            }

        } else {

            double prev_energy = energy;
            energy = compute_energy_LP_higher_order(Q);

            if (abs(energy - prev_energy) < prox_tol) ++count;
            else count = 0;
            if (count >= 5) {
                //std::cout << "\n##CONV: energy - prev_energy < " << prox_tol << " for last " << count
                //    << " iterations! terminating...\n";
                break;
            }
            if( energy < best_energy) {
                best_Q = Q;
                best_energy = energy;
            }
        }
        if (params.less_confident_percent > 0) {
            float confidence_tol = params.confidence_tol;
            std::vector<int> pI;
            less_confident_pixels(pI, best_Q, confidence_tol);
            double percent = double(pI.size())/double(Q.cols())*100;
            if (percent > params.less_confident_percent) {
                //std::cout << "\n##CONV: Less confident pixels are greater than " << params.less_confident_percent
                //    << "%, terminating...\n";
                break;
            }
        }

    } while(it<maxiter);

    return best_Q;
}



void DenseCRF::computeUCondGrad(MatrixXf & Us, const MatrixXf & Q) const {
    //assert(Us.cols() == N_);
    //assert(Us.rows() == M_);
    Us.fill(0);
    int min,max,min_lab,max_lab,min_ind=0,max_ind=0, min_lab_ind=0,max_lab_ind=0;
    int sum = 0;
    for(int reg = 0; reg < R_; reg++) {
        min = std::numeric_limits<int>::max();
        max = std::numeric_limits<int>::min();
        for (int lab = 0; lab < M_; lab++) {
            for (auto it : super_pixel_container_[reg]) { //loop through all of the pixels in this super pixel
                //we want to find the smallest and largest values, we then set the conditional gradient accordingly
                if (min > Q(lab,it)) {
                    min_ind = it;
                    min_lab_ind = lab;
                    min = Q(lab,it);
                }
                if (max < Q(lab,it)) {
                    max_ind = it;
                    max_lab_ind = lab;
                    max = Q(lab,it);
                }
            }
        }
        Us(min_lab_ind,min_ind) = constants_[reg] * exp_of_superpixels_(reg);
        Us(max_lab_ind,max_ind) = -constants_[reg] * exp_of_superpixels_(reg);              
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



double DenseCRF::assignment_energy_higher_order(const VectorXs & l) const {
    VectorXf unary = unaryEnergy(l);
    VectorXf pairwise = pairwise_energy_true(l);

    VectorXf total_energy = unary + pairwise;

    assert( total_energy.rows() == N_);
    double ass_energy = 0;
    for( int i=0; i< N_; ++i) {
        ass_energy += total_energy[i];
    }

    double ho_energy = 0.0;
    short prev = 0;
    for (int reg = 0; reg < R_; reg++) {
        prev = l[super_pixel_container_[reg][0]];
        for (int pix = 0; pix < super_pixel_container_[reg].size(); pix++) {
            if (prev != l[pix]) {
                ass_energy += exp_of_superpixels_[reg];
                break;
            } 
        }
        ho_energy += (R_ - 1) * exp_of_superpixels_[reg];
    }

    return ass_energy + ho_energy;
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


double DenseCRF::compute_energy_higher_order(const MatrixXf & Q) const {
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

    //super pixels
    MatrixXf constant = exp_of_superpixels_.replicate( 1, M_).transpose();
    energy += (z_labels_.cwiseProduct(constant).sum() + multiplySuperPixels((MatrixXf::Ones(M_,R_) - z_labels_).cwiseProduct(constant),(Q - MatrixXf::Ones(M_,N_))));

    return energy;
}

double DenseCRF::compute_energy_LP_higher_order(const MatrixXf & Q) const {
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

    for (int reg = 0; reg < R_; reg++)
    {
        MatrixXf Q_p;
        for (const auto & it : super_pixel_container_[reg])
        {
            Q_p.conservativeResize(M_, Q_p.cols() + 1);
            Q_p.col(Q_p.cols() - 1) = Q.col(it);
        }

        VectorXf maxVals = Q_p.rowwise().maxCoeff();
        VectorXf minVals = Q_p.rowwise().minCoeff();
        VectorXf maxDiff = maxVals - minVals;
        float maxDiff_l = maxDiff.maxCoeff();
        energy += exp_of_superpixels_[reg] * maxDiff_l;
    }

    return energy;
}







