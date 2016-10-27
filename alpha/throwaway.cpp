#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include "densecrf.h"
#include "file_storage.hpp"
#include "inference.hpp"
#include "newton_cccp.hpp"

void test_writing(){

    MatrixXf test = MatrixXf::Identity(10, 12);

    save_matrix("test.csv", test);

    MatrixXf loaded = load_matrix("test.csv");

    int diff = test.cwiseNotEqual(loaded).count();

    assert(diff == 0);

    std::cout << test  << '\n';
    std::cout  << '\n';
    std::cout << loaded << '\n';

}


void create_junk_parameters(){
    VectorXf junk_param = VectorXf::Constant(469, 1);

    save_matrix("parameters.csv", junk_param);
}



void initialize_crf_parameters(){
    img_size size;
    size.width= 320;
    size.height = 213;

    int M = 21;

    unsigned char * img = load_image("/data/MSRC/Images_by_partition/test/1_27_s.bmp", size);

    DenseCRF2D crf(size.height, size.width, M);
    // Add simple pairwise potts terms

    crf.addPairwiseGaussian( 1, 1, new MatrixCompatibility(-MatrixXf::Identity(M,M)));
    // Add a longer range label compatibility term
    crf.addPairwiseBilateral(1,1,1,1,1, img, new MatrixCompatibility(-10 * MatrixXf::Identity(M,M) ));

    // The kernel parameters can't be initialized directly because teh addPairwise function pre-computes the features
    VectorXf kp(7);
    kp<< 3,3,80,80,13,13,13;
    crf.setKernelParameters(kp);


    VectorXf params(M *(M+1) + 7);

    int pos=0;
    int pairwise_size = crf.labelCompatibilityParameters().rows();
    VectorXf pair_params = crf.labelCompatibilityParameters();
    for (int i=0; i < pairwise_size; i++) {
        params(pos+i) = pair_params(i);
    }
    pos += pairwise_size;

    int kernel_size = crf.kernelParameters().rows();
    VectorXf kernel_params = crf.kernelParameters();
    for (int i=0; i < kernel_size; i++) {
        params(pos+i) = kernel_params(i);
    }

    save_matrix("parameters.csv", params);
}


void test_cccp(){

    VectorXf initial_proba(5);
    initial_proba << 0.2, 0.3, 0.4, 0.05, 0.05;

    float lbda = 3;

    VectorXf cste(5);
    cste << 2, 3, 4, 5, 6;

    VectorXf state(6);
    state.head(5) = initial_proba;
    state(5) = 1;

    newton_cccp(state, cste, lbda);



}

void matrix_eigenvalues(){
    MatrixXf test(4,4);
    test<< 1, 2, 3, 4,
        4, 3, 2, 1,
        2, 2, 4, 4,
        2, 2, 4, 4;

    std::cout << test.eigenvalues() << '\n';

}

void test_label_matrix_loading(){
    std::string  path_to_labels = "/data/MSRC/GT_by_partition/val/4_11_s_GT.bmp";
    label_matrix lbl = load_label_matrix(path_to_labels, "MSRC");
}


void test_pascal_loading(){
    std::string dataset_name = "Pascal2010";
    Dataset ds = get_dataset_by_name(dataset_name);
    std::vector<std::string> all_images = ds.get_all_split_files("Train");
    std::string output_directory = "/data/tests/";

    std::string image_name = all_images[1];

    std::string path_to_image = ds.get_image_path(image_name);
    std::string path_to_unaries = ds.get_unaries_path(image_name);
    std::string path_to_output = get_output_path(output_directory, image_name);
    std::string path_to_parameters = "/data/densecrf/alpha/learned_parameters.csv";

    // minimize_mean_field(path_to_image, path_to_unaries,
    //                     path_to_output, path_to_parameters);


    img_size size;
    MatrixXf un = load_unary(path_to_unaries, size);


    std::cout << un.rows() << '\n';
    std::cout << un.cols() << '\n';
    unaries_baseline(path_to_unaries, path_to_output, dataset_name);

}

void check_pairwise_energies(){
    std::string dataset_name = "Pascal2010";
    Dataset ds = get_dataset_by_name(dataset_name);
    std::vector<std::string> all_images = ds.get_all_split_files("Train");
    std::string output_directory = "/data/tests/";

    std::string image_name = all_images[1];

    std::string path_to_image = ds.get_image_path(image_name);
    std::string path_to_unaries = ds.get_unaries_path(image_name);
    std::string path_to_output = get_output_path(output_directory, image_name);
    std::string path_to_parameters = "/data/densecrf/alpha/learned_parameters.csv";

    img_size size;
    MatrixXf un = load_unary(path_to_unaries, size);
    un = 0* un;
    unsigned char * img = load_image(path_to_image, size);

    // Load a crf
    DenseCRF2D crf(size.width, size.height, un.rows());
    Potts_weight_set parameters(3, 3, 50, 15, 5);
    crf.setUnaryEnergy(un);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             img, new PottsCompatibility(parameters.bilat_potts_weight));



    //  Use a labeling that is uniform.
    MatrixXf Q_un = MatrixXf::Zero(un.rows(), size.width * size.height);
    for (int i = 0; i < Q_un.cols(); i++) {
        Q_un(0, i) = 1;
    }
    // Use a labeling that is not uniform.
    MatrixXf Q_rd = MatrixXf::Zero(un.rows(), size.width * size.height);
    for (int i =0; i < Q_rd.cols(); i++) {
        Q_rd( i % Q_rd.rows(), i) = 1;
    }
    std::cout << "Our way: " << '\n';
    std::cout << "Uniform labeling energy: "<< crf.compute_energy(Q_un) << '\n'; // -1.47465e+06
    std::cout << "Heterogeneous labeling energy: " << crf.compute_energy(Q_rd)<< '\n'; // -70977.7

    std::cout << "Old way: " << '\n';
    std::cout << "Uniform labeling pairwise energy: " << crf.pairwiseEnergy(crf.currentMap(Q_un)).sum() << '\n';
    std::cout << "Heterogeneous labeling pairwise energy: " << crf.pairwiseEnergy(crf.currentMap(Q_rd)).sum() << '\n';

}

void eigen_test() {
    std::srand(1337); // Set the seed
	MatrixXf m(3,4);
    m << 1, 2, 3, 4,
        4, 3, 2, 1,
        2, 2, 4, 4;
	VectorXf v(4);
	std::cout << "#m#\n" << m << std::endl;
	v = m.row(0);
	std::cout << "#v#\n" << v << std::endl;
	m.row(2) = v;
	std::cout << "#m#\n" << m << std::endl;
	std::cout << "#v#\n" << v << std::endl;
	VectorXf v1 = m.colwise().sum();
	std::cout << "#v1#\n" << v1 << std::endl;

	MatrixXf sm(3, 4);
	MatrixXi ind(3, 4);
	sortRows(m, ind);
	for(int i = 0; i < 4; ++i) {
		for (int j= 0; j < 3; ++j) {
			sm(j, i) = m(j, ind(j, i));
		}
	}
	std::cout << "#sm#\n" << sm << std::endl;

    MatrixXf q(3,5), qq(3,5);
    for(int i = 0; i < q.cols(); ++i) {
		for (int j= 0; j < q.rows(); ++j) {
			q(j, i) = std::rand() % 10;
		}
	}
    rescale(qq, q);
    std::cout << "#q#\n" << q << std::endl;
    std::cout << "#qq#\n" << qq << std::endl;
}

void ph_test() {        
    std::srand(1337); // Set the seed
    int s = 5;
    split_array* values = new split_array[ s ];
    split_array* new_values = new split_array[ s ];
    
    for (int i = 0; i < s; ++i) {
        split_array* sp = values + i;
        for (float& v : sp->data) v = std::rand()%100;
    }
	memset(new_values, 0, s*sizeof(split_array));
    
    std::cout << "\n#values#\n";
    for (int i = 0; i < s; ++i) {
        split_array* sp = values + i;
        for (float v : sp->data) std::cout << v << ", ";
        std::cout << std::endl;
    }
    std::cout << "\n#new_values#\n";
    for (int i = 0; i < s; ++i) {
        split_array* sp = new_values + i;
        for (float v : sp->data) std::cout << v << ", ";
        std::cout << std::endl;
    }

    split_array* tmp = new split_array[ s ];
    delete[] tmp;
    bool* t = new bool[10];
    delete[] t;

    memcpy(new_values, values, s*sizeof(split_array));
    std::cout << "\nafter-memcpy\n";

    std::cout << "\n#values#\n";
    for (int i = 0; i < s; ++i) {
        split_array* sp = values + i;
        for (float v : sp->data) std::cout << v << ", ";
        std::cout << std::endl;
    }
    std::cout << "\n#new_values#\n";
    for (int i = 0; i < s; ++i) {
        split_array* sp = new_values + i;
        for (float v : sp->data) std::cout << v << ", ";
        std::cout << std::endl;
    }
}

// make a step of qp_gamma: -- \cite{NNQP solver Xiao and Chen 2014} - O(n) implementation!!
void qp_gamma_step_test(VectorXf & v_gamma, const VectorXf & v_pos_h, const VectorXf & v_neg_h, const float qp_delta, 
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
void qp_gamma_multiplyC_test(VectorXf & v_out, const VectorXf & v_in, const int M, const float lambda) {
    // C: lambda*I - (lambda/m)*ones
    float sum = v_in.sum();
    v_out.fill(sum);
    v_out = (-1.0/float(M)) * v_out + v_in;
    v_out *= lambda;
}

void qp_gamma_test() {
    int M_ = 3;
    float lambda = 0.1;
    MatrixXf C(M_, M_), neg_C(M_, M_), pos_C(M_, M_), abs_C(M_, M_);
    VectorXf v_gamma(M_), v_y(M_), v_pos_h(M_), v_neg_h(M_), v_step(M_), v_tmp(M_), v_tmp2(M_);

    pos_C = MatrixXf::Identity(M_, M_) * (1-1.0/float(M_));				      
    pos_C *= lambda;	// pos_C
    neg_C = MatrixXf::Ones(M_, M_) - MatrixXf::Identity(M_, M_);	
    neg_C /= float(M_);													      
    neg_C *= lambda; 	// neg_C
    C = pos_C - neg_C;	// C
    abs_C = pos_C + neg_C;	// abs_C
    
    std::srand(13);
    for(int i = 0; i < M_; ++i) {
        v_gamma(i) = std::rand()%10;
    }

    std::cout << "\n#v_gamma\n" << v_gamma << std::endl;
    std::cout << "\n#C\n" << C << std::endl;
    v_tmp = C * v_gamma;
    std::cout << "\n#C*v_gamma (old)\n" << v_tmp << std::endl;
    qp_gamma_multiplyC_test(v_tmp2, v_gamma, M_, lambda);
    std::cout << "\n#C*v_gamma (new)\n" << v_tmp2 << std::endl;
    
}

int main(int argc, char *argv[])
{
    //eigen_test();
    qp_gamma_test();
    return 0;
}

