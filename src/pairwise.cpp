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
#include "pairwise.h"
#include <iostream>

Kernel::~Kernel() {
}
class DenseKernel: public Kernel {
protected:
	NormalizationType ntype_;
	KernelType ktype_;
	Permutohedral lattice_;
	VectorXf norm_;
	MatrixXf f_;
	MatrixXf parameters_;
	void initLattice( const MatrixXf & f, int max_size=-1 );
    void merge(Kernel & other, MatrixXf const & features, bool overlap) {
    	// The normallizations won't be correct
    	assert(ntype_==NO_NORMALIZATION);
    	const MatrixXf & f = other.features();
    	if(overlap) {
    		lattice_.add( f.rightCols(f.cols()-1) );
    	} else {
    		lattice_.add( f );
    	}
    	f_ = features;
    }
	void filter( MatrixXf & out, const MatrixXf & in, bool transpose ) const {
		// Read in the values
		if( ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && !transpose) || (ntype_ == NORMALIZE_AFTER && transpose))
			out = in*norm_.asDiagonal();
		else
			out = in;
	
		// Filter
		if( transpose )
			lattice_.compute( out, out, true );
		else
			lattice_.compute( out, out );
// 			lattice_.compute( out.data(), out.data(), out.rows() );
	
		// Normalize again
		if( ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && transpose) || (ntype_ == NORMALIZE_AFTER && !transpose))
			out = out*norm_.asDiagonal();
	}
	void filter_lower_left( MatrixXf & out, int middle_low, int middle_high ) const {
		// Normalization makes no sense here since this would always return 1
	
		// Filter
			lattice_.compute_lower_left( out, middle_low, middle_high );
	}
	// Compute d/df a^T*K*b
	MatrixXf kernelGradient( const MatrixXf & a, const MatrixXf & b ) const {
		MatrixXf g = 0*f_;
		lattice_.gradient( g.data(), a.data(), b.data(), a.rows() );
		return g;
	}
	MatrixXf featureGradient( const MatrixXf & a, const MatrixXf & b ) const {
		if (ntype_ == NO_NORMALIZATION )
			return kernelGradient( a, b );
		else if (ntype_ == NORMALIZE_SYMMETRIC ) {
			MatrixXf fa = lattice_.compute( a*norm_.asDiagonal(), true );
			MatrixXf fb = lattice_.compute( b*norm_.asDiagonal() );
			MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
			VectorXf norm3 = norm_.array()*norm_.array()*norm_.array();
			MatrixXf r = kernelGradient( 0.5*( a.array()*fb.array() + fa.array()*b.array() ).matrix()*norm3.asDiagonal(), ones );
			return - r + kernelGradient( a*norm_.asDiagonal(), b*norm_.asDiagonal() );
		}
		else if (ntype_ == NORMALIZE_AFTER ) {
			MatrixXf fb = lattice_.compute( b );
			
			MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
			VectorXf norm2 = norm_.array()*norm_.array();
			MatrixXf r = kernelGradient( ( a.array()*fb.array() ).matrix()*norm2.asDiagonal(), ones );
			return - r + kernelGradient( a*norm_.asDiagonal(), b );
		}
		else /*if (ntype_ == NORMALIZE_BEFORE )*/ {
			MatrixXf fa = lattice_.compute( a, true );
			
			MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
			VectorXf norm2 = norm_.array()*norm_.array();
			MatrixXf r = kernelGradient( ( fa.array()*b.array() ).matrix()*norm2.asDiagonal(), ones );
			return -r+kernelGradient( a, b*norm_.asDiagonal() );
		}
	}
public:
	DenseKernel(const MatrixXf & f, KernelType ktype, NormalizationType ntype, int max_size):f_(f), ktype_(ktype), ntype_(ntype) {
		if (ktype_ == DIAG_KERNEL)
			parameters_ = VectorXf::Ones( f.rows() );
		else if( ktype == FULL_KERNEL )
			parameters_ = MatrixXf::Identity( f.rows(), f.rows() );
		initLattice( f, max_size );
	}
	virtual void apply_lower_left( MatrixXf & out, int middle_low, int middle_high) const {
		filter_lower_left(out, middle_low, middle_high);
	}
	virtual void apply( MatrixXf & out, const MatrixXf & Q ) const {
		filter( out, Q, false );
	}
	virtual void applyTranspose( MatrixXf & out, const MatrixXf & Q ) const {
		filter( out, Q, true );
	}
	virtual VectorXf parameters() const {
		if (ktype_ == CONST_KERNEL)
			return VectorXf();
		else if (ktype_ == DIAG_KERNEL)
			return parameters_;
		else {
			MatrixXf p = parameters_;
			p.resize( p.cols()*p.rows(), 1 );
			return p;
		}
	}
	virtual void setParameters( const VectorXf & p ) {
		if (ktype_ == DIAG_KERNEL) {
			parameters_ = p;
			initLattice( p.asDiagonal() * f_ );
		}
		else if (ktype_ == FULL_KERNEL) {
			MatrixXf tmp = p;
			tmp.resize( parameters_.rows(), parameters_.cols() );
			parameters_ = tmp;
			initLattice( tmp * f_ );
		}
	}
	virtual VectorXf gradient( const MatrixXf & a, const MatrixXf & b ) const {
		if (ktype_ == CONST_KERNEL)
			return VectorXf();
		MatrixXf fg = featureGradient( a, b );
		if (ktype_ == DIAG_KERNEL)
			return (f_.array()*fg.array()).rowwise().sum();
		else {
			MatrixXf p = fg*f_.transpose();
			p.resize( p.cols()*p.rows(), 1 );
			return p;
		}
	}

	virtual MatrixXf features() const {
		return f_;
	}

	virtual KernelType ktype() const {
		return ktype_;
	}

	virtual NormalizationType ntype() const {
		return ntype_;
	}
};

void DenseKernel::initLattice( const MatrixXf & f, int max_size ) {
	const int N = f.cols();
	lattice_.init( f, max_size );
	
	
	if ( ntype_ != NO_NORMALIZATION ) {
		norm_ = lattice_.compute( VectorXf::Ones( N ).transpose() ).transpose();

		if ( ntype_ == NORMALIZE_SYMMETRIC ) {
		for ( int i=0; i<N; i++ )
				norm_[i] = 1.0 / sqrt(norm_[i]+1e-20);
		}
		else {
			for ( int i=0; i<N; i++ )
				norm_[i] = 1.0 / (norm_[i]+1e-20);
		}
	}
}

PairwisePotential::~PairwisePotential(){
	delete compatibility_;
	delete kernel_;
}
PairwisePotential::PairwisePotential(const MatrixXf & features, LabelCompatibility * compatibility, KernelType ktype, NormalizationType ntype, int max_size) : compatibility_(compatibility) {
	kernel_ = new DenseKernel( features, ktype, ntype, max_size );
}
void PairwisePotential::merge(PairwisePotential & other, MatrixXf const & features, bool overlap) {
	assert(compatibility_->parameters()(0) == other.parameters()(0));
	kernel_->merge(*other.getKernel(), features, overlap);
}
void PairwisePotential::apply(MatrixXf & out, const MatrixXf & Q) const {
	kernel_->apply( out, Q );
	
	// Apply the compatibility
	compatibility_->apply( out, out );
}
void PairwisePotential::applyTranspose(MatrixXf & out, const MatrixXf & Q) const {
	kernel_->applyTranspose( out, Q );
	// Apply the compatibility
	compatibility_->applyTranspose( out, out );
}
void PairwisePotential::apply_lower(MatrixXf & out, const MatrixXi & ind) const {
	MatrixXf const & features = kernel_->features();
	MatrixXf sorted_features = features;
	MatrixXf single_label_out(1, features.cols());

	for(int label=0; label<ind.rows(); ++label) {
		// Sort the features with the scores for this label
		for(int j=0; j<features.cols(); ++j) {
			sorted_features.col(j) = features.col(ind(label, j));
		}

		single_label_out.fill(0);
		// Create a new lattice with these features
		/*PairwisePotential sorted_pairwise(
			sorted_features,
			new PottsCompatibility(1),
			CONST_KERNEL,
			NO_NORMALIZATION
		);
		sorted_pairwise.apply_lower_sorted(single_label_out);
		*/
		PairwisePotential* p = apply_lower_sorted_merge(single_label_out, features, features.cols());
		delete p;
		out.row(label) = single_label_out;
	}
	compatibility_->apply(out, out);
}
PairwisePotential* PairwisePotential::apply_lower_sorted_merge(
		MatrixXf & out,
		MatrixXf const & features,
		int max_size ) const {
	int size = out.cols();

	if(size <= 0) {
		// This should never happen, this would create an empty permutohedral
		assert(false);
	} else if(size<=10) {
		// Alpha is a magic scaling constant (write Rudy if you really wanna understand this)
		double alpha = 1.0 / 0.6;
		for(int c=0; c<out.cols(); ++c)
			out(0, c) = 0;
		for(int c=0; c<out.cols(); ++c) {
            for(int b=0; b<c; ++b) {
                VectorXf featDiff = (features.col(c) - features.col(b));
                out(0, c) += exp(-featDiff.squaredNorm()) * alpha;
            }
        }

		PairwisePotential* pairwise = new PairwisePotential(
			features,
			new PottsCompatibility(compatibility_->parameters()(0)),
			CONST_KERNEL,
			NO_NORMALIZATION,
			max_size
		);
		return pairwise;
	} else {
		int middle_low, middle_high;
		bool overlap = false;
		if(size%2==0) {
			middle_low = size/2;
			middle_high = size/2;
		} else if(size%2==1) {
			middle_low = floor(size/2.0);
			middle_high = floor(size/2.0) + 1;
			overlap = true;
		}

		MatrixXf out_tmp(1,middle_high);
		out_tmp.fill(0);
		PairwisePotential* upper_pairwise = apply_lower_sorted_merge(out_tmp, features.leftCols(middle_high), max_size);
		out.leftCols(middle_high) += out_tmp;

		out_tmp.fill(0);
		PairwisePotential* lower_pairwise = apply_lower_sorted_merge(out_tmp, features.rightCols(middle_high), middle_high);
		out.rightCols(middle_high) += out_tmp;

		upper_pairwise->merge(*lower_pairwise, features, overlap);
		delete lower_pairwise;

		upper_pairwise->getKernel()->apply_lower_left(out, middle_low, middle_high);

		return upper_pairwise;
	}
}
void PairwisePotential::apply_lower_sorted(MatrixXf & out) const {
	MatrixXf const & features = kernel_->features();
	int size = out.cols();

	if(size <= 1) {
		// Only a=b term remaining
		return;
	} else if(size<=10) {
		// Alpha is a magic scaling constant (write Rudy if you really wanna understand this)
		double alpha = 1.0 / 0.6;
		for(int c=0; c<out.cols(); ++c)
			out(0, c) = 0;
		for(int c=0; c<out.cols(); ++c) {
            for(int b=0; b<c; ++b) {
                VectorXf featDiff = (features.col(c) - features.col(b));
                out(0, c) += exp(-featDiff.squaredNorm()) * alpha;
            }
        }
	} else {
		int middle_low, middle_high;
		if(size%2==0) {
			middle_low = size/2;
			middle_high = size/2;
		} else if(size%2==1) {
			middle_low = floor(size/2.0);
			middle_high = floor(size/2.0) + 1;
		}

		// Compute the lower left part
		kernel_->apply_lower_left(out, middle_low, middle_high);

		// Compute the upper left
		PairwisePotential upper_pairwise(
			features.leftCols(middle_high),
			new PottsCompatibility(compatibility_->parameters()(0)),
			CONST_KERNEL,
			NO_NORMALIZATION
		);
		MatrixXf upper_out(1,middle_high);
		upper_out.fill(0);
		upper_pairwise.apply_lower_sorted(upper_out);
		out.leftCols(middle_high) += upper_out;

		// Compute the lower right
		PairwisePotential lower_pairwise(
			features.rightCols(middle_high),
			new PottsCompatibility(compatibility_->parameters()(0)),
			CONST_KERNEL,
			NO_NORMALIZATION
		);
		MatrixXf lower_out(1,middle_high);
		lower_out.fill(0);
		lower_pairwise.apply_lower_sorted(lower_out);
		out.rightCols(middle_high) += lower_out;
	}
}
VectorXf PairwisePotential::parameters() const {
	return compatibility_->parameters();
}
void PairwisePotential::setParameters( const VectorXf & v ) {
	compatibility_->setParameters( v );
}
VectorXf PairwisePotential::gradient( const MatrixXf & b, const MatrixXf & Q ) const {
	MatrixXf filtered_Q = 0*Q;
	// You could reuse the filtered_b from applyTranspose
	kernel_->apply( filtered_Q, Q );
	return compatibility_->gradient(b,filtered_Q);
}
VectorXf PairwisePotential::kernelParameters() const {
	return kernel_->parameters();
}
void PairwisePotential::setKernelParameters( const VectorXf & v ) {
	kernel_->setParameters( v );
}
VectorXf PairwisePotential::kernelGradient( const MatrixXf & b, const MatrixXf & Q ) const {
	MatrixXf lbl_Q = 0*Q;
	// You could reuse the filtered_b from applyTranspose
	compatibility_->apply( lbl_Q, Q );
	return kernel_->gradient(b,lbl_Q);
}
MatrixXf PairwisePotential::features() const {
	return kernel_->features();
}
KernelType PairwisePotential::ktype() const {
	return kernel_->ktype();
}
NormalizationType PairwisePotential::ntype() const {
	return kernel_->ntype();
}
Kernel* PairwisePotential::getKernel() const {
	return kernel_;
}
MatrixXf PairwisePotential::compatibility_matrix(int nb_labels) const {
	return compatibility_->matrixForm(nb_labels);
}
