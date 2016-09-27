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
#include "objective.h"
#include "pairwise.h"
#include <vector>
#include <utility>

typedef std::pair<double, double> perf_measure;

void expAndNormalize( MatrixXf & out, const MatrixXf & in);
void sumAndNormalize( MatrixXf & out, const MatrixXf & in, const MatrixXf & Q);
void normalize(MatrixXf & out, const MatrixXf & in);

class LP_inf_params {
public: 
	float prox_reg_const;	// proximal regularization constant
	float dual_gap_tol;		// dual gap tolerance
	int prox_max_iter;		// maximum proximal iterations
	int fw_max_iter;		// maximum FW iterations
	int qp_max_iter;		// maximum qp-gamma iterations
	float qp_tol;			// qp-gamma tolerance
	float qp_const;			// const used in qp-gamma
    bool best_int;          // return the Q that yields the best integer energy
	LP_inf_params(float prox_reg_const, float dual_gap_tol, int prox_max_iter, 
			int fw_max_iter, int qp_max_iter, float qp_tol, float qp_const, bool best_int);
	LP_inf_params();	// default values
};


/**** DenseCRF ****/
class DenseCRF{
protected:
	// Number of variables and labels
	int N_, M_;

	// Store the unary term
	UnaryEnergy * unary_;

	// Store all pairwise potentials
	std::vector<PairwisePotential*> pairwise_;

	// How to stop inference
	bool compute_kl = false;

	// Don't copy this object, bad stuff will happen
	DenseCRF( DenseCRF & o ){}
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

    // set potts compatibility weight such that ratio *  unary enegy = pairwise energy (given Q)
	void setPairwisePottsWeight(float ratio, const MatrixXf & Q);

	// Set the unary potential (ownership will be transfered to this class)
	void setUnaryEnergy( UnaryEnergy * unary );
	// Add a constant unary term
	void setUnaryEnergy( const MatrixXf & unary );
	// Add a logistic unary term
	void setUnaryEnergy( const MatrixXf & L, const MatrixXf & f );
	UnaryEnergy* getUnaryEnergy();


	MatrixXf unary_init() const;
	MatrixXf uniform_init() const;
	// Run inference and return the probabilities
	MatrixXf inference(const MatrixXf & init,  int n_iterations ) const;
	MatrixXf inference(const MatrixXf & init) const;
	std::vector<perf_measure> tracing_inference(MatrixXf & init, double time_limit = 0) const;

	// Run the energy minimisation on the QP
	// First one is the Lafferty-Ravikumar version of the QP
	MatrixXf qp_inference(const MatrixXf & init) const;
	MatrixXf qp_inference(const MatrixXf & init, int nb_iterations) const;
	std::vector<perf_measure> tracing_qp_inference(MatrixXf & init, double time_limit = 0) const;
	// Second one is the straight up QP, using CCCP to be able to optimise shit up.
    MatrixXf qp_cccp_inference(const MatrixXf & init) const;
	std::vector<perf_measure> tracing_qp_cccp_inference(MatrixXf & init, double time_limit =0) const;
	// Third one the QP-cccp defined in the Krahenbuhl paper, restricted to concave label compatibility function.
	MatrixXf concave_qp_cccp_inference(const MatrixXf & init) const;
	std::vector<perf_measure> tracing_concave_qp_cccp_inference(MatrixXf & init, double time_limit = 0) const;
	// Run the energy minimisation on the LP
    MatrixXf lp_inference(MatrixXf & init, bool use_cond_grad) const;
    MatrixXf lp_inference_new(MatrixXf & init) const;
    MatrixXf lp_inference_prox(MatrixXf & init, LP_inf_params & params) const;
	std::vector<perf_measure> tracing_lp_inference(MatrixXf & init, bool use_cond_grad, double time_limit = 0) const;

	// compare permutohedral and bruteforce energies (testing code only)
    void compare_energies(const MatrixXf & Q, double & ph_energy, double & bf_energy, bool qp=true, bool ph_old = false) const;

	// Perform the rounding based on argmaxes
	MatrixXf max_rounding(const MatrixXf & estimates) const;
	// Perform randomized roundings
	MatrixXf interval_rounding(const MatrixXf & estimates, int nb_random_rounding = 10) const;

	// Run the inference with gradually lower lambda values.
	MatrixXf grad_inference(const MatrixXf & init) const;

	// Run the inference with cccp optimization
	MatrixXf cccp_inference(const MatrixXf & init) const;

	// Run MAP inference and return the map for each pixel
	VectorXs map( int n_iterations ) const;

	// Step by step inference
	MatrixXf startInference() const;
	void stepInference( MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2 ) const;
	VectorXs currentMap( const MatrixXf & Q ) const;

	// Learning functions
	// Compute the gradient of the objective function over mean-field marginals with
	// respect to the model parameters
	double gradient( int n_iterations, const ObjectiveFunction & objective, VectorXf * unary_grad, VectorXf * lbl_cmp_grad, VectorXf * kernel_grad=NULL ) const;
public: /* Debugging functions */
	// Compute the unary energy of an assignment l
	VectorXf unaryEnergy( const VectorXs & l ) const;

	// Compute the pairwise energy of an assignment l (half of each pairwise potential is added to each of it's endpoints)
	VectorXf pairwiseEnergy( const VectorXs & l, int term=-1 ) const;

    // compute true pairwise energy for LP objective given integer labelling
	VectorXf pairwise_energy_true( const VectorXs & l, int term=-1 ) const;

	// Compute the energy of an assignment l.
	double assignment_energy( const VectorXs & l) const;

    // Compute the true energy of an assignment l -- actual energy (differs by a const to assignment_energy - in pairwise case)
	double assignment_energy_true( const VectorXs & l) const;

	// Compute the KL-divergence of a set of marginals
	double klDivergence( const MatrixXf & Q ) const;

    // Compute the energy associated with the QP relaxation (const - true)
    double compute_energy( const MatrixXf & Q) const;

    // Compute the true-energy associated with the QP relaxation
    double compute_energy_true( const MatrixXf & Q) const;

    // Compute the true-energy associated with the QP relaxation - bf
    double compute_energy_true( const MatrixXf & Q, PairwisePotential** no_norm_pairwise, int nb_pairwise) const;

    // Compute the energy associated with the LP relaxation
    double compute_energy_LP(const MatrixXf & Q, PairwisePotential** no_norm_pairwise, int nb_pairwise) const;

    // Compute the energy associated with the LP relaxation
    double compute_energy_LP(const MatrixXf & Q) const;

	// Compute the value of a Lafferty-Ravikumar QP
	double compute_LR_QP_value(const MatrixXf & Q, const MatrixXf & diag_dom) const;

public: /* Parameters */
	void compute_kl_divergence();
	VectorXf unaryParameters() const;
	void setUnaryParameters( const VectorXf & v );
	VectorXf labelCompatibilityParameters() const;
	void setLabelCompatibilityParameters( const VectorXf & v );
	VectorXf kernelParameters() const;
	void setKernelParameters( const VectorXf & v );
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
	
	// Set the unary potential for a specific variable
	using DenseCRF::setUnaryEnergy;
};
