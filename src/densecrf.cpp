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
#include <set>

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
MatrixXf DenseCRF::unary_init() const {
    MatrixXf Q;
    expAndNormalize(Q, unary_->get());
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
        desc(M_,N_), psis(M_,N_), sx(M_,N_);
    MatrixP temp_dot(M_,N_);

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

    energy = compute_LR_QP_value(Q, diag_dom);
    int num_iter=0;
    while( (old_energy - energy) > 100){
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
        // - \frac{\theta'^T(s-x) + 2 x^T \psi (s-x)}{ 2 * (s-x)^T \psi (s-x) }
        sx = desc - Q;
        psis.fill(0);
        for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply(tmp, sx);
            psis += tmp;
        }
        psis += diag_dom.cwiseProduct(sx);
        double num =  2 * dotProduct(Q, psis, temp_dot) + dotProduct(unary, sx, temp_dot);
        // Num should be negative, otherwise our choice of s was necessarily wrong.
        double denom = dotProduct(sx, psis, temp_dot);
        // Denom should be positive, otherwise our choice of psi was not convex enough.

        double optimal_step_size = - num / (2 * denom);
        if (optimal_step_size > 1) {
            optimal_step_size = 1;
        }
        if (denom == 0) {
            // This means that the conditional gradient is the same
            // than the current step and we have converged.
            optimal_step_size = 0;
        }

        // Take a step
        Q += optimal_step_size * sx;
        energy = compute_LR_QP_value(Q, diag_dom);
    }
    return Q;
}

MatrixXf DenseCRF::qp_cccp_inference(const MatrixXf & init) const {
    MatrixXf Q(M_, N_), Q_old(M_,N_), grad(M_,N_), unary(M_, N_), tmp(M_, N_),
        desc(M_, N_), sx(M_, N_),  psis(M_, N_), psix(M_,N_), diag_dom(M_,N_);
    MatrixP temp_dot(M_,N_);
    // Compute the smallest eigenvalues, that we need to make bigger
    // than 0, to ensure that the problem is convex.
    MatrixXf full_ones = -MatrixXf::Ones(M_, N_);
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

    // Get initial estimates
    // Initialize state to the unaries
    // Warning: We don't get exactly the same optimum depending on the initialisation
    Q = init;
    // Q.fill(1/ (float) M_);

    // Compute the value of the energy
    double old_energy;
    double energy = compute_energy(Q);

    do {
        // New value of the linearization point.
        old_energy = energy;
        Q_old = Q;

        double convex_energy = energy + dotProduct(Q, diag_dom.cwiseProduct(2*Q_old - Q), temp_dot);
        double old_convex_energy;


        int convex_rounds = 0;
        double optimal_step_size = 1;
        do {
            old_convex_energy = convex_energy;
            // Compute gradient of the convex problem
            psix.fill(0);
            for( unsigned int k=0; k<pairwise_.size(); k++ ) {
                pairwise_[k]->apply( tmp, Q);
                psix += tmp;
            }

            grad = unary +
                2 * psix +
                2 * diag_dom.cwiseProduct(Q_old - Q);

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

            double num = dotProduct(unary, sx, temp_dot) +
                2 * dotProduct(Q, psis, temp_dot) +
                2 * dotProduct(sx, diag_dom.cwiseProduct(Q-Q_old), temp_dot);
            assert(num<=0); // This is negative if desc is really the good minimizer

            double denom = dotProduct(desc, psis, temp_dot) +
                dotProduct(desc, diag_dom.cwiseProduct(desc), temp_dot); // (s-x)d(s-x)
            assert(denom>0); // This is positive if we did our decomposition correctly

            double cst = dotProduct(unary, Q, temp_dot) +
                dotProduct(Q, psix, temp_dot) +
                dotProduct(Q-2*Q_old, diag_dom.cwiseProduct(Q), temp_dot);

            optimal_step_size = - num/ (2 *denom);

            if (optimal_step_size > 1) {
                optimal_step_size = 1;
            }

            // std::cout << "Step size: " << optimal_step_size << '\n';

            Q += optimal_step_size * sx;

            // std::cout << "Coefficients: "<< denom << '\t' << num << '\t' << cst << '\n';
            convex_energy = pow(optimal_step_size, 2) * denom + optimal_step_size * num + cst;

            // energy = compute_energy(Q);
            // old_convex_energy = energy + lambda_eig * dotProduct(Q, 2*Q_old -Q, temp_dot);
            // std::cout << old_convex_energy  << '\n';
            // std::cout << convex_energy << '\n';

            assert(valid_probability(Q));
            convex_rounds++;
        } while ( (old_convex_energy - convex_energy) > 100 && convex_rounds<3 && optimal_step_size != 0);
        // We are now (almost) at a minimum of the convexified problem, so we
        // stop solving the convex problem and get a new convex approximation.

        // Compute our current value of the energy;
        energy = compute_energy(Q);
    } while ( (old_energy -energy) > 100);
    return Q;
}

void add_noise(MatrixXf & Q, float var) {
    Q += MatrixXf::Random(Q.rows(), Q.cols())*var;
    for(int col=0; col<Q.cols(); ++col) {
        Q.col(col) /= Q.col(col).sum();
    }
}

struct classcomp {
  bool operator() (const VectorXf& lhs, const VectorXf& rhs) const
  {
    for(int i=0; i<lhs.rows(); ++i)
    {
        if(lhs(i)!=rhs(i)){
            return true;
        }
    }
    return false;
  }
};

MatrixXf DenseCRF::lp_inference(MatrixXf & init) const {
    MatrixXf Q(M_, N_), ones(M_, N_), base_grad(M_, N_), tmp(M_, N_), unary(M_, N_),
            tmp2(M_, N_), grad(M_, N_);
    MatrixXi ind(M_, N_);
    VectorXi K(N_);
    VectorXd sum(N_);
    float noise_var = 1e-6;

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

    // Get parameters
    unary.fill(0);
    if(unary_){
        unary = unary_->get();
    }

    Q = init;


    // Compute the value of the energy
    double old_energy;
    assert(valid_probability(Q));
    double energy = compute_energy_LP(Q, no_norm_pairwise, nb_pairwise);
    std::cout << "0: " << energy << "\n";
    //std::cout << ((Q.array()-Q.mean()).array()*(Q.array()-Q.mean()).array()).mean() << "\n";
    //std::cout<<Q.rightCols(5).topRows(5)<<std::endl;

    // precompute the constant part of the gradient
    base_grad = unary;
    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        // Add the full sum
        no_norm_pairwise[k]->apply(tmp, ones);
        base_grad -= tmp;

        // Remove the diagonal terms
        base_grad = base_grad.array() - (1*no_norm_pairwise[k]->parameters()(0));
    }

    int it=0;
    do {
        std::set<VectorXf, classcomp> unique_pixels;
        VectorXf pix;
        for(int col=0; col<Q.cols(); ++col) {
            pix = Q.col(col);
            unique_pixels.insert(pix);
        }
        std::cout << "Different pixels: " << unique_pixels.size() << "\n";


        ++it;
        old_energy = energy;
        
        // Grad start with constant part
        grad = base_grad;

        // Add changing part
        //add_noise(Q, noise_var);
        
        sortRows(Q, ind);

        for( unsigned int k=0; k<nb_pairwise; k++ ) {
            no_norm_pairwise[k]->apply_lower(tmp, ind);
            tmp2.fill(0);
            for(i=0; i<tmp.cols(); ++i) {
                for(j=0; j<tmp.rows(); ++j) {
                    tmp2(j, ind(j, i)) = tmp(j, i);
                }
            }
            grad += 2*tmp2;
        }

        // Sub-gradient descent step
        float lr = 1.0/(100000+it);
        Q -= lr*grad;

        // Project solution

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
                tmp(k, i) = std::max(Q(k, i) - sum(i)/K(i), (double)0);
            }
        }
        Q = tmp;

        assert(valid_probability(Q));
        energy = compute_energy_LP(Q, no_norm_pairwise, nb_pairwise);
        std::cout << it << ": " << energy << "\n";
        //std::cout << ((Q.array()-Q.mean()).array()*(Q.array()-Q.mean()).array()).mean() << "\n";
        //std::cout<<Q.rightCols(5).topRows(5)<<std::endl;
    } while(it<5);//fabs(old_energy -energy) > 1e-5);
    std::cout <<"final: " << energy << "\n";
    std::set<VectorXf, classcomp> unique_pixels;
    VectorXf pix;
    for(int col=0; col<Q.cols(); ++col) {
        pix = Q.col(col);
        unique_pixels.insert(pix);
    }
    std::cout << "Different pixels: " << unique_pixels.size() << "\n";


    free(no_norm_pairwise);
    return Q;
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
    MatrixXf tmp;
    for( unsigned int k=0; k<nb_pairwise; k++ ) {
        // Remove once the full
        no_norm_pairwise[k]->apply( tmp, Q );
        energy += tmp.sum();
        // Add the diagonal
        energy += Q.rows()*(Q*no_norm_pairwise[k]->parameters()(0)).sum();
        // Add twice the lower matrix
        no_norm_pairwise[k]->apply_lower(tmp, ind);
        energy -= 2*dotProduct(Q, tmp, dot_tmp);;
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
