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
#include "objective.h"
#include "pairwise.h"
#include <vector>
#include <utility>

typedef std::pair<double, double> perf_measure;

void expAndNormalize( MatrixXf & out, const MatrixXf & in);
void sumAndNormalize( MatrixXf & out, const MatrixXf & in, const MatrixXf & Q);
void normalize(MatrixXf & out, const MatrixXf & in);

void get_limited_indices(MatrixXf const & Q, std::vector<int> & indices);
MatrixXf get_restricted_matrix(MatrixXf const & in, std::vector<int> const & indices);
MatrixXf get_extended_matrix(MatrixXf const & in, std::vector<int> const & indices, int max_rows);
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

	//store the super pixel term
	Matrix<float, Dynamic, Dynamic> super_pixel_classifier_;
	std::vector<std::vector<double>> super_pixel_container_; //is a vector of super pixels which holds a vector of pixel locations for that super pixel

	// Store all pairwise potentials
	std::vector<PairwisePotential*> pairwise_;

	// Store all pairwise potentials -- no-normalization: used to caluclate energy 
	std::vector<PairwisePotential*> no_norm_pairwise_;

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
	//Add a super pixel term
	int setSuperPixelEnergy( const unsigned char * img);
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

	//===============================================================
	//The following definitions are for Tom Joys 4YP which computes the QP with a non convex function and with super pixel terms
	MatrixXf qp_inference_non_convex(const MatrixXf & init) const;
	MatrixXf qp_inference_super_pixels(const MatrixXf & init, double K = 100000) const;
	MatrixXf qp_inference_super_pixels_non_convex(const MatrixXf & init, double K = 100000) const;
	std::vector<perf_measure> tracing_qp_inference_non_convex(MatrixXf & init, double time_limit = 0) const;
	std::vector<perf_measure> tracing_qp_inference_super_pixels_non_convex(MatrixXf & init, double time_limit = 0) const;
	//===============================================================

	// Second one is the straight up QP, using CCCP to be able to optimise shit up.
    MatrixXf qp_cccp_inference(const MatrixXf & init) const;
	std::vector<perf_measure> tracing_qp_cccp_inference(MatrixXf & init, double time_limit =0) const;
	// Third one the QP-cccp defined in the Krahenbuhl paper, restricted to concave label compatibility function.
	MatrixXf concave_qp_cccp_inference(const MatrixXf & init) const;
	std::vector<perf_measure> tracing_concave_qp_cccp_inference(MatrixXf & init, double time_limit = 0) const;
	// Run the energy minimisation on the LP
    MatrixXf lp_inference(MatrixXf & init, bool use_cond_grad, bool full_mat = false) const;
    MatrixXf lp_inference_new(MatrixXf & init) const;
    MatrixXf lp_inference_prox(MatrixXf & init, LP_inf_params & params) const;
    MatrixXf lp_inference_prox_super_pixels(MatrixXf & init, LP_inf_params & params) const;
    MatrixXf lp_inference_prox_restricted(MatrixXf & init, LP_inf_params & params) const;
	std::vector<perf_measure> tracing_lp_inference(MatrixXf & init, bool use_cond_grad, double time_limit = 0, bool full_mat = false) const;
	std::vector<perf_measure> tracing_lp_inference_prox(MatrixXf & init, LP_inf_params & params, 
            double time_limit = 0, std::string out_file_name = "") const;
    std::vector<perf_measure> tracing_lp_inference_prox_restricted(MatrixXf & init, LP_inf_params & params, 
            double time_limit = 0) const;

	// compare permutohedral and bruteforce energies (testing code only)
    void compare_energies(MatrixXf & Q, double & ph_energy, double & bf_energy, 
        bool qp=true, bool ph_old = false, bool subgrad = false) const;

	// compare old (_dc) and new (_ord) permutohedral implementation for timing
    std::vector<perf_measure> compare_lpsubgrad_timings(MatrixXf & Q, bool cmp_subgrad = false) const;

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

    // KL-divergence between two probabilities KL(Q||P)
    double klDivergence(const MatrixXf & Q, const MatrixXf & P) const;

    // Compute the energy associated with the QP relaxation (const - true)
    double compute_energy( const MatrixXf & Q) const;

    // Compute the energy associated with the QP-CCCP relaxation (const - true)
    double compute_energy_CCCP( const MatrixXf & Q) const;

    // Compute the true-energy associated with the QP relaxation
    double compute_energy_true( const MatrixXf & Q) const;

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
	
	//add a super pixel term, this function computes the super pixels using edison mean-shift algorithm
	void addSuperPixel(unsigned char * img, int spatial_radius = 8, int range_radius = 4, int min_region_count = 2500, SpeedUpLevel = NO_SPEEDUP);


	// Set the unary potential for a specific variable
	using DenseCRF::setUnaryEnergy;
};
