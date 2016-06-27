#include <vector>
#include <string>
#include "file_storage.hpp"
#include "inference.hpp"
#include "densecrf.h"
#include <iostream>
#include <fstream>

#define NUM_LABELS 16

MatrixXf get_unaries(const unsigned char * left_img, const unsigned char * right_img, img_size & size){
    MatrixXf unaries(NUM_LABELS, size.height * size.width);
    for (int off=0; off<NUM_LABELS; off++) {
        for (int j = 0; j < size.height; j++) {
            for (int i=0; i < size.width; i++) {
                float diff = 0;
                if (i + off < size.width) {// No penalty if we can't see the corresponding thing?
                    diff += abs(right_img[(i+j*size.width)*3+0] - left_img[(i+off+j*size.width)*3+0]);
                    diff += abs(right_img[(i+j*size.width)*3+1] - left_img[(i+off+j*size.width)*3+1]);
                    diff += abs(right_img[(i+j*size.width)*3+2] - left_img[(i+off+j*size.width)*3+2]);
                }
                unaries(off, i + j* size.width) = diff;
            }
        }
    }

    return unaries;
}

MatrixXf get_unaries_from_file(std::string path, img_size & size) {
    std::ifstream txt_file(path.c_str());
    int nbr_var, nbr_label, M, m;
    txt_file >> nbr_var >> nbr_label;
    txt_file >> M;
    txt_file >> m;
    MatrixXf unaries(nbr_label, size.height * size.width);
    assert(nbr_var == size.height * size.width);

    float val;
    for(int i=0; i<nbr_var; ++i) {
        int col = i/size.height;
        int row = i%size.height;
        int index = row*size.width + col;
        for(int l=0; l<nbr_label; ++l) {
            txt_file >> val;
            unaries(l, index) = val;
        }
    }

    txt_file.close();
    return unaries;
}

void write_down_perf2(double timing, double final_energy, double rounded_energy, std::string path_to_output){
    std::string txt_output = path_to_output;
    txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");

    std::ofstream txt_file(txt_output.c_str());
    txt_file << timing << '\t' << final_energy << '\t' << rounded_energy << std::endl;
    txt_file.close();
}

int main(int argc, char *argv[])
{
    if(argc < 3){
        std::cout << "./stereo path_to_stereo_folder method" << '\n';
        std::cout << "Example: ./stereo /data/Stereo/tsukuba/ qpcccp" << '\n';
        return 1;
    }

    std::string stereo_folder = argv[1];
    std::string method = argv[2];
    std::string left_image_path = stereo_folder + "imL.png";
    std::string right_image_path = stereo_folder + "imR.png";
    std::string output_image_path = stereo_folder + "out_" + method + ".bmp";
    std::string unary_path = stereo_folder + "unary.txt";

    Potts_weight_set parameters(5, 50, 2, 15, 50);

    img_size size = {-1, -1};

    unsigned char * left_img = load_image(left_image_path, size);
    unsigned char * right_img = load_image(right_image_path, size);


    // MatrixXf unaries = get_unaries(left_img, right_img, size);
    MatrixXf unaries = get_unaries_from_file(unary_path, size);

    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             left_img, new PottsCompatibility(parameters.bilat_potts_weight));

    clock_t start, end;
    start = clock();
    MatrixXf Q = crf.unary_init();
    std::vector<perf_measure> traced_perfs;
    std::vector<perf_measure> new_perfs;
    double time_budget = 200;
    if (method == "mf5") {
        Q = crf.inference(Q, 5);
    } else if (method == "mf") {
        Q = crf.inference(Q);
    } else if (method == "lrqp") {
        Q = crf.qp_inference(Q);
    } else if (method == "qpcccp") {
        Q = crf.qp_inference(Q);
        Q = crf.qp_cccp_inference(Q);
    } else if (method == "proper_qpcccp_ccv"){
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);
    } else if (method == "ccv"){
        Q = crf.concave_qp_cccp_inference(Q);
    } else if (method == "sg_lp"){
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);
        Q = crf.lp_inference(Q, false);
    } else if (method == "cg_lp"){
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);
        Q = crf.lp_inference(Q, true);
    } else if (method == "unary"){
        (void)0;
    } else if (method == "qp5"){
        Q = crf.qp_inference(Q, 5);
    } else if (method == "tracing-qp"){
        traced_perfs = crf.tracing_qp_inference(Q, time_budget);
    } else if (method == "tracing-only-ccv") {
        traced_perfs = crf.tracing_concave_qp_cccp_inference(Q, time_budget);
    } else if (method == "tracing-mf"){
        traced_perfs = crf.tracing_inference(Q, time_budget);
    } else if (method == "tracing-qpcccp") {
        traced_perfs = crf.tracing_qp_inference(Q);
        for (int i = 0; i < traced_perfs.size(); i++) {
            time_budget -= traced_perfs[i].first;
        }
        new_perfs = crf.tracing_qp_cccp_inference(Q, time_budget);
        traced_perfs.insert( traced_perfs.end(), new_perfs.begin(), new_perfs.end() );
    } else if (method == "tracing-proper_qpcccp_cv"){
        traced_perfs = crf.tracing_qp_inference(Q);
        for (int i = 0; i < traced_perfs.size(); i++) {
            time_budget -= traced_perfs[i].first;
        }
        new_perfs = crf.tracing_concave_qp_cccp_inference(Q, time_budget);
        traced_perfs.insert( traced_perfs.end(), new_perfs.begin(), new_perfs.end() );
    } else if (method == "tracing-sg_lp"){
        traced_perfs = crf.tracing_qp_inference(Q);
        new_perfs = crf.tracing_concave_qp_cccp_inference(Q);
        traced_perfs.insert( traced_perfs.end(), new_perfs.begin(), new_perfs.end());
        for (int i = 0; i < traced_perfs.size(); i++) {
            time_budget -= traced_perfs[i].first;
        }
        new_perfs = crf.tracing_lp_inference(Q, false, time_budget);
        traced_perfs.insert( traced_perfs.end(), new_perfs.begin(), new_perfs.end());
    } else{
        std::cout << method << '\n';
        std::cout << "Unrecognised method.\n Proper error handling would do something but I won't." << '\n';
    }

    end = clock();

    if(method.find("tracing")!=std::string::npos){
        std::string txt_output = output_image_path;
        txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
        std::ofstream txt_file(txt_output);
        for (int it=0; it<traced_perfs.size(); it++) {
            txt_file << it << '\t' << traced_perfs[it].first << '\t' << traced_perfs[it].second << std::endl;
        }
        txt_file.close();
    } else {
        double timing = (double(end-start)/CLOCKS_PER_SEC);
        double final_energy = crf.compute_energy(Q);
        double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
        std::cout << "Fractional Energy: " << final_energy << '\n';
        std::cout << "Integer Energy: " << discretized_energy << '\n';

        write_down_perf2(timing, final_energy, discretized_energy, output_image_path);
        save_map(Q, size, output_image_path, "Stereo_special");
    }
}
