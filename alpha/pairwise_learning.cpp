#include "optimization.h"
#include "densecrf.h"
#include "file_storage.hpp"
#include <iostream>
#include <omp.h>

#define NUM_LABELS 21

class DenseCRFEnergy: public EnergyFunction {

protected:
    VectorXf initial_lbl_param, initial_knl_param; // Our CRF's parameter
    // Doesn't contain the standard deviation parameters.

    VectorXf all_parameters;

    int nb_iter; // Do we want to limit the numbers of MeanFields iteration????

    bool train_pairwise_, train_kernel_;

    float l2_norm_ = 0;

    std::vector<std::string> all_images_;
    std::string path_to_dataset_;

public:
    DenseCRFEnergy(std::vector<std::string> all_images, std::string path_to_dataset, int NIT,
                   bool train_pairwise=true, bool train_kernel=true):nb_iter(NIT),
                                                                     train_pairwise_(train_pairwise),train_kernel_(train_kernel),
                                                                     l2_norm_(0.0f), all_images_(all_images), path_to_dataset_(path_to_dataset){
    }

    void setParametersValue(VectorXf parameters){
        all_parameters = parameters;
    }

    virtual VectorXf initialValue() {
        return all_parameters;
    }




    virtual double single_image_gradient(const std::string & path_to_image, const std::string & path_to_unaries,
                                         const std::string & path_to_label, const VectorXf & parameters, VectorXf & gradient){

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
        crf.addPairwiseGaussian( 1, 1, new MatrixCompatibility( MatrixXf::Identity(M,M)));
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


        Hamming objective(labeling, 0.0);

        VectorXf dl = 0*initial_lbl_param, dk = 0*initial_knl_param; // Note: why was dl initialized to du size?
        double r = crf.gradient( nb_iter, objective, NULL, train_pairwise_?&dl:NULL, train_kernel_?&dk:NULL );
        gradient.resize( train_pairwise_*pairwise_size + train_kernel_*kernel_size);
        gradient << -(train_pairwise_?dl:VectorXf()), -(train_kernel_?dk:VectorXf());
        r = -r;

        if( l2_norm_ > 0 ) {
            gradient += l2_norm_ * parameters;
            r += 0.5*l2_norm_ * (parameters.dot(parameters));
        }
        return r;
    }



    virtual double gradient( const VectorXf & x, VectorXf & dx) {
        std::cout << "\n\n\n\nStarting a global gradient computation" << '\n';
        std::string latest_parameters = "latest_parameters.csv";
        save_matrix(latest_parameters, x);


        std::vector<double> losses;
        std::vector<VectorXf> gradients;

#pragma omp parallel for
        for(int i=0; i< all_images_.size(); ++i){
            std::string image_name = all_images_[i];
            std::cout << image_name << '\n';
            std::string unaries_path = get_unaries_path(path_to_dataset_, image_name);
            std::string image_path = get_image_path(path_to_dataset_, image_name);
            std::string gt_path = get_ground_truth_path(path_to_dataset_, image_name);

            VectorXf img_gradient = VectorXf(dx.size());
            double img_loss = single_image_gradient(image_path, unaries_path, gt_path, x, img_gradient);

            losses.push_back(img_loss);
            gradients.push_back(img_gradient);
        }

        double loss = 0;
        dx = 0 * dx;
        for (int i=0; i < losses.size(); i++) {
            loss += losses[i];
            dx = dx + gradients[i];
        }
        return loss;
    }

    void setL2Norm(float norm){
        l2_norm_ = norm;
    }


};


int main(int argc, char *argv[])
{
    if (argc<3) {
        std::cout << "learn_pairwise split path_to_dataset" << '\n';
        std::cout << "Example: ./learn_pairwise Train /data/MSRC/" << '\n';
        return 1;
    }


    std::string dataset_split = argv[1];
    std::string path_to_dataset = argv[2];

    std::string init_parameters = "parameters.csv";
    std::string learned_parameters = "learned_parameters.csv";
    std::vector<std::string> split_images = get_all_split_files(path_to_dataset, dataset_split);

    // initialize parameters
    VectorXf parameters;
    if(file_exist(init_parameters)){
        parameters = load_matrix(init_parameters);
    } else {
        std::cout << "Initialize some parameters first" << '\n';
        return 1;
    }

    DenseCRFEnergy to_optimize(split_images, path_to_dataset, 5);
    to_optimize.setParametersValue(parameters);

    int nb_restart = 2;
    int verbose = true;
    VectorXf p = minimizeLBFGS(to_optimize, nb_restart, verbose);

    // Write-down parameters
    save_matrix(learned_parameters, parameters);

    return 0;
}
