#include "optimization.h"
#include "densecrf.h"
#include "file_storage.hpp"
#include <iostream>

#define NUM_LABELS 21

class DenseCRFEnergy: public EnergyFunction {

protected:
    VectorXf unary_param, initial_lbl_param, initial_knl_param; // Our CRF's parameter
    // Doesn't contain the standard deviation parameters.

    DenseCRF & crf_;
    const ObjectiveFunction & objective_;
    int nb_iter; // Do we want to limit the numbers of MeanFields iteration????

    bool train_pairwise_, train_kernel_;

    float l2_norm_ = 0;

public:
    DenseCRFEnergy(DenseCRF & crf, const ObjectiveFunction & objective, int NIT, bool train_pairwise=true, bool train_kernel=true):crf_(crf), objective_(objective),nb_iter(NIT),train_pairwise_(train_pairwise),train_kernel_(train_kernel),l2_norm_(0.0f){
        unary_param = crf_.unaryParameters();
        initial_lbl_param = crf_.labelCompatibilityParameters();
        initial_knl_param = crf_.kernelParameters();
    }

    virtual VectorXf initialValue() {
        // Adjust the size of the parameters vector containing everything
        VectorXf p(train_pairwise_*initial_lbl_param.rows() + train_kernel_ * initial_knl_param.rows());
        p << (train_pairwise_?initial_lbl_param:VectorXf()), (train_kernel_?initial_knl_param:VectorXf());
        return p;
    }

    virtual double gradient( const VectorXf & x, VectorXf & dx ) {
        int p = 0;
        if (train_pairwise_) {
            crf_.setLabelCompatibilityParameters( x.segment( p, initial_lbl_param.rows() ) );
            p += initial_lbl_param.rows();
        }
        if (train_kernel_)
            crf_.setKernelParameters( x.segment( p, initial_knl_param.rows() ) );

        VectorXf dl = 0*initial_lbl_param, dk = 0*initial_knl_param; // Note: why was dl initialized to du size?
        double r = crf_.gradient( nb_iter, objective_, NULL, train_pairwise_?&dl:NULL, train_kernel_?&dk:NULL );
        dx.resize( train_pairwise_*dl.rows() + train_kernel_*dk.rows() );
        dx << -(train_pairwise_?dl:VectorXf()), -(train_kernel_?dk:VectorXf());
        r = -r;

        if( l2_norm_ > 0 ) {
            dx += l2_norm_ * x;
            r += 0.5*l2_norm_ * (x.dot(x));
        }
        return r;
    }

    void setL2Norm(float norm){
        l2_norm_ = norm;
    }


};

void training_crf(const std::string & path_to_image, const std::string & path_to_unaries,
                  const std::string & path_to_label, bool train_pairwise, bool train_kernel,
                  VectorXf & parameters){

    img_size size;
    MatrixXf unaries = load_unary(path_to_unaries, size);

    int M = unaries.rows();

    unsigned char * img = load_image(path_to_image, size);

    VectorXs labeling = load_labeling(path_to_label, size);

    // Setup the CRF model
    DenseCRF2D crf(size.width, size.height, unaries.rows());
    // Add a logistic unary term
    crf.setUnaryEnergy(unaries);

    // We can't really modify the parameters of the weights between
    // the various gaussians because they are not modelled in the
    // code, they are folded into the

    // We put all the values of the standard deviation to 1 so that
    // the parameters learned from the optimization make direct sense

    // Add simple pairwise potts terms
    crf.addPairwiseGaussian( 1, 1, new MatrixCompatibility(MatrixXf::Identity(M,M)));
    // Add a longer range label compatibility term
    crf.addPairwiseBilateral( 1, 1, 1, 1, 1, img, new MatrixCompatibility( MatrixXf::Identity(M,M) ));

    // Setting the crf to where we're at in the learning.
    int pos=0;
    int pairwise_size = crf.labelCompatibilityParameters().rows();
    crf.setLabelCompatibilityParameters(parameters.segment(pos, pairwise_size));
    pos += pairwise_size;
    int kernel_size = crf.kernelParameters().rows();
    crf.setKernelParameters(parameters.segment(pos, kernel_size));
    pos += kernel_size;


    assert(pos == parameters.size());


    IntersectionOverUnion objective(labeling);

    int nb_iter = 5;

    DenseCRFEnergy energy(crf, objective, nb_iter, train_pairwise, train_kernel);
    //energy.setL2Norm(1e-3);

    bool verbose = true;
    VectorXf optimized = minimizeLBFGS(energy, 2, verbose );

    parameters = optimized;

}


int main(int argc, char *argv[])
{
    if (argc<3) {
        std::cout << "learn_pairwise split path_to_dataset" << '\n';
        std::cout << "Example: ./learn_pairwise Train /home/rudy/datasets/MSRC/" << '\n';
        return 1;
    }


    std::string dataset_split = argv[1];
    std::string path_to_dataset = argv[2];

    std::string init_parameters = "parameters.csv";
    std::string learned_parameters = "learned_parameters.csv";
    std::string last_learned_parameters = "last_learned_parameters.csv";

    std::vector<std::string> split_images = get_all_split_files(path_to_dataset, dataset_split);

    // initialize parameters
    VectorXf parameters;
    if(file_exist(init_parameters)){
        parameters = load_matrix(init_parameters);
    } else {
        // Poor initialization, should be improved.
        int nb_parameters = 2 +      // spatial Gaussian std-dev
            NUM_LABELS * (NUM_LABELS + 1) + // Label compatibility for both the spatial and bilateral
            5;                      // Bilateral Gaussian std-dev
        parameters = VectorXf::Constant(nb_parameters, 1);
    }

    for(int i=0; i< split_images.size(); ++i){
        std::string image_name = split_images[i];
        std::cout << image_name << '\n';
        std::string unaries_path = get_unaries_path(path_to_dataset, image_name);
        std::string image_path = get_image_path(path_to_dataset, image_name);
        std::string gt_path = get_ground_truth_path(path_to_dataset, image_name);

        training_crf(image_path, unaries_path, gt_path, true, true, parameters);

        save_matrix(last_learned_parameters, parameters);

        // ToDo: piecewise training
    }

// Write-down parameters
    save_matrix(learned_parameters, parameters);


    return 0;
}
