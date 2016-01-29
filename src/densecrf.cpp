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
#include <cmath>
#include <cstring>
#include <iostream>

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

MatrixXf DenseCRF::inference ( int n_iterations ) const {
	MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2;
	unary.fill(0);
	if( unary_ ){
		unary = unary_->get();
	}
	expAndNormalize( Q, -unary );

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

MatrixXf DenseCRF::inference () const {
	MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2, old_Q(M_, N_);
	float old_kl, kl;
	unary.fill(0);
	if( unary_ ){
		unary = unary_->get();
	}
	expAndNormalize( Q, -unary );

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
	std::cout << "Nb of needed iterations: " << count << '\n';
	return Q;
}

MatrixXf DenseCRF::qp_inference() const {
    // Todo: We don't get always decreasing value, which is weird and
    // shouldn't happen
	MatrixXf Q(M_, N_), unary(M_, N_), diag_dom(M_,N_), tmp(M_,N_), grad(M_, N_),
        desc(M_,N_), psis(M_,N_), sx(M_,N_);

    // Get initial estimates
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }
    // Initialize state to the unaries
    expAndNormalize(Q, -unary);

    // Build proxy unaries for the added terms
    // Compute the dominant diagonal

    // Note: All the terms in the pairwise matrix are negatives
    // so to get the sum of the abs value, you need to get the
    // product with the matrix full of -1.
    MatrixXf full_ones = -MatrixXf::Ones(M_, N_);
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, full_ones);
        diag_dom += tmp;
    }
    // Update the proxy_unaries
    unary = unary - diag_dom;

    // Compute the value of the energy
    double old_energy = std::numeric_limits<double>::max();
    double energy;

    energy = compute_LR_QP_value(Q, diag_dom);

    while( (old_energy - energy) > 1e-3){
        old_energy = energy;
        // Compute the gradient at the current estimates.
        grad = unary;
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply( tmp, Q);
            grad += 2 *tmp;
        }
        grad += 2 * diag_dom.cwiseProduct(Q);

        // Get a Descent direction by minimising < \nabla E, s >
        descent_direction(desc, grad);

        // Solve for the best step size. The best step size is
        // - \frac{x^T \Psi (s-x) + 0.5 \phi (s-x)}{(s-x)^T \Psi (s-x)}
        sx = desc - Q;
        psis.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx);
            psis += tmp;
        }
        psis += diag_dom.cwiseProduct(sx);
        double num =  2 * Q.cwiseProduct(psis).sum() + unary.cwiseProduct(sx).sum();
        double denom = 2* sx.cwiseProduct(psis).sum();
        double optimal_step_size = - num / denom;
        if (optimal_step_size > 1) {
            optimal_step_size = 1;
        }

        // Take a step
        Q += optimal_step_size * sx;
        energy = compute_LR_QP_value(Q, diag_dom);
    }
    std::cout << compute_energy(Q) << '\n';
    return Q;
}

MatrixXf DenseCRF::qp_cccp_inference() const {
    MatrixXf Q(M_, N_);
    // Compute the largest eigenvalues necessary to make sure that the problem is convex.

    // TODO: Write down on paper the updates rules for cccp, should just be
    // a simpler version than the ones for KL-divergence minimisation


    // Get initial estimates

    // CCCP loop start, while energy is decreasing

    // Compute the linearisation of the concave part

    // Solve the convex minimisation problem, using conditional gradient descent
    // This should be really easy to reuse some of the functions in the relaxed qp.

    // CCCP loop end

    return Q;
}


MatrixXf DenseCRF::cccp_inference() const {
    MatrixXf Q( M_, N_), tmp1, unary(M_, N_), tmp2, old_Q(M_, N_);
    float old_kl, kl;
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    // Compute the largest eigenvalues
    float lambda_eig = 0;
    for (int i=0; i<pairwise_.size(); i++) {
        lambda_eig += pick_lambda_eig(pairwise_[i]->compatibility_matrix(M_));
    }
    expAndNormalize(Q, -unary);

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
    std::cout << "Nb of needed iterations: " << count << '\n';
    return Q;
}


MatrixXf DenseCRF::grad_inference() const {
    MatrixXf Q( M_, N_ ), tmp1, unary( M_, N_ ), tmp2, old_Q(M_, N_), Q_prev_lambda(M_, N_);
    unary.fill(0);
    if( unary_ ) {
        unary = unary_->get();
    }
    expAndNormalize( Q, -unary );

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
    MatrixXf Q = inference( n_iterations );
    // Find the map
    return currentMap( Q );
}

double DenseCRF::assignment_energy( const VectorXs & l) const {
    VectorXf unary = unaryEnergy(l);
    VectorXf pairwise = pairwiseEnergy(l);

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
    energy += unary.cwiseProduct(Q).sum();
    energy -= diag_dom.cwiseProduct(Q).sum();

    // Add all pairwise terms
    MatrixXf tmp;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q );
        energy += Q.cwiseProduct(tmp).sum();
    }
    energy += Q.cwiseProduct(diag_dom.cwiseProduct(Q)).sum();
    return energy;
}


double DenseCRF::compute_energy(const MatrixXf & Q) const {
    double energy = 0;
    // Add the unary term
    if( unary_ ) {
        MatrixXf unary = unary_->get();
        for( int i=0; i<Q.cols(); i++ )
            for( int l=0; l<Q.rows(); l++ )
                energy += unary(l,i)*Q(l,i);
    }
    // Add all pairwise terms
    MatrixXf tmp;
    for( unsigned int k=0; k<pairwise_.size(); k++ ) {
        pairwise_[k]->apply( tmp, Q );
        energy += (Q.array()*tmp.array()).sum();
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
