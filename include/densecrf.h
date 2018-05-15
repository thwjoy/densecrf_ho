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

#pragma once
#include "unary.h"
#include "labelcompatibility.h"
#include "msImageProcessor.h"
#include "pairwise.h"
#include <vector>
#include <utility>

typedef Matrix<short,Dynamic,1> VectorXs;

typedef std::pair<double, double> perf_measure;

void expAndNormalize( MatrixXf & out, const MatrixXf & in);
void less_confident_pixels(std::vector<int> & indices, const MatrixXf & Q, float tol = 0.99);

class LP_inf_params {
public: 
	float prox_reg_const;	// proximal regularization constant
	float dual_gap_tol;		// dual gap tolerance
	float prox_energy_tol;	// proximal energy tolerance
	int prox_max_iter;		// maximum proximal iterations
	int fw_max_iter;		// maximum FW iterations
	int qp_max_iter;		// maximum qp-gamma iterations
	float qp_tol;			// qp-gamma tolerance
	float qp_const;			// const used in qp-gamma
    bool best_int;          // return the Q that yields the best integer energy
    bool accel_prox;        // accelerated proximal method
    // multi-plane FW params (work_set_size==0 && approx_fw_iter==0 ==> standard FW)
    int work_set_size;      // working set size 
    int approx_fw_iter;     // approximate FW iterations
    // less-confident switch
    float less_confident_percent;   // percentage of less confident pixels to break
    float confidence_tol;           // tolerance to decide a pixel to be less-confident
	LP_inf_params(float prox_reg_const, float dual_gap_tol, float prox_energy_tol, 
        int prox_max_iter, int fw_max_iter, 
        int qp_max_iter, float qp_tol, float qp_const, 
        bool best_int, bool accel_prox, 
        int work_set_size, int approx_fw_iter, 
        float less_confident_percent, float confident_tol);
	LP_inf_params();	// default values
    LP_inf_params(const LP_inf_params& params); // copy constructor
};


/**** DenseCRF ****/
class DenseCRF{
protected:
	// Number of variables, labels and super pixel terms
	int N_, M_, R_;

	VectorXf mean_of_superpixels_;
	VectorXf exp_of_superpixels_;

	// Store the unary term
	UnaryEnergy * unary_;

    //store the z labels
	Matrix<float, Dynamic, Dynamic> z_labels_;

	std::vector<std::vector<double>> super_pixel_container_; //is a vector of super pixels which holds a vector of pixel locations for that super pixel
	std::vector<int> pixel_to_regions_;

	// Store all pairwise potentials
	std::vector<PairwisePotential*> pairwise_;

	//store the constants for the clique
	std::vector<float> constants_;

	// Store all pairwise potentials -- no-normalization: used to caluclate energy 
	std::vector<PairwisePotential*> no_norm_pairwise_;

	// How to stop inference
	bool compute_kl = false;

	// Don't copy this object, bad stuff will happen
	DenseCRF( DenseCRF & o ){}


	float multiplySuperPixels(const MatrixXf & p1, const MatrixXf & p2) const;
	float multiplyDecompSuperPixels(const MatrixXf & p1, const MatrixXf & p2, int reg) const;
	MatrixXf multiplySuperPixels(const MatrixXf & p) const;
	MatrixXf multiplyDecompSuperPixels(const MatrixXf & p, int reg) const; //used for the DCneg case
	void computeUCondGrad(MatrixXf & Us, const MatrixXf & Q) const;

public:
	// Create a dense CRF model of size N with M labels
	DenseCRF( int N, int M );
	virtual ~DenseCRF();

	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	// (ownership of LabelCompatibility will be transfered to this class)
	virtual void addPairwiseEnergy( const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );

	// Add your own favorite pairwise potential (ownership will be transfered to this class)
	void addPairwiseEnergy( PairwisePotential* potential );
	// Set the unary potential (ownership will be transfered to this class)
	void setUnaryEnergy( UnaryEnergy * unary );
	// Add a constant unary term
	void setUnaryEnergy( const MatrixXf & unary );
	// Add a logistic unary term
	void setUnaryEnergy( const MatrixXf & L, const MatrixXf & f );
	//Add a super pixel term
	int setSuperPixelEnergy( const unsigned char * img);
	UnaryEnergy* getUnaryEnergy();

	MatrixXf unary_init() const;

    const MatrixXf qp_inference_super_pixels_non_convex(const MatrixXf & init);
	// Run the energy minimisation on the LP
    const MatrixXf lp_inference_prox_super_pixels(const MatrixXf & init, const LP_inf_params & params) const;

	// Perform the rounding based on argmaxes
    MatrixXf max_rounding(const MatrixXf & estimates) const;

    VectorXs currentMap( const MatrixXf & Q ) const;


public: /* Debugging functions */
	// Compute the unary energy of an assignment l
	VectorXf unaryEnergy( const VectorXs & l ) const;

    // compute true pairwise energy for LP objective given integer labelling
	VectorXf pairwise_energy_true( const VectorXs & l, int term=-1 ) const;

	// Compute the energy of an assignment l.
    double assignment_energy_higher_order( const VectorXs & l) const;

    // Compute the true energy of an assignment l -- actual energy (differs by a const to assignment_energy - in pairwise case)
	double assignment_energy_true( const VectorXs & l) const;

    // Compute the true-energy associated with the QP relaxation
    double compute_energy_higher_order( const MatrixXf & Q) const;

    // Compute the energy associated with the LP relaxation
    double compute_energy_LP_higher_order(const MatrixXf & Q) const;

	// Compute the value of a Lafferty-Ravikumar QP
    double compute_LR_QP_value(const MatrixXf & Q, const MatrixXf & diag_dom) const;

};

class DenseCRF2D:public DenseCRF{
protected:
	// Width, height of the 2d grid
	int W_, H_;

public:
	// Create a 2d dense CRF model of size W x H with M labels
	DenseCRF2D( int W, int H, int M );
	virtual ~DenseCRF2D();
	// Add a Gaussian pairwise potential with standard deviation sx and sy
	void addPairwiseGaussian( float sx, float sy, LabelCompatibility * function=NULL, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );
	
	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	void addPairwiseBilateral( float sx, float sy, float sr, float sg, float sb, const unsigned char * im, LabelCompatibility * function=NULL, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );
	
	//add a super pixel term, this function computes the super pixels using edison mean-shift algorithm
	void addSuperPixel(std::string path_to_classifier,unsigned char * img, float constant, float normaliser);

    void addSuperPixel(unsigned char * img, int spatial_radius, int range_radius, int min_region_count, float constant, float normaliser);

	// Set the unary potential for a specific variable
	using DenseCRF::setUnaryEnergy;
};
