#include <vector>
#include <string>
#include "file_storage.hpp"
#include "inference.hpp"
#include "densecrf.h"
#include <iostream>

#define NUM_LABELS 16

MatrixXf get_unaries(const unsigned char * left_img, const unsigned char * right_img, img_size & size){
    MatrixXf unaries(NUM_LABELS, size.height * size.width);
    for (int off=0; off<NUM_LABELS; off++) {
        for (int j = 0; j < size.height; j++) {
            for (int i=0; i < size.width; i++) {
                float diff = 0;
                if (i + off < size.width) {// No penalty if we can't see the corresponding thing?
                    diff += abs(left_img[(i+j*size.width)*3+0] - right_img[(i+off+j*size.width)*3+0]);
                    diff += abs(left_img[(i+j*size.width)*3+1] - right_img[(i+off+j*size.width)*3+1]);
                    diff += abs(left_img[(i+j*size.width)*3+2] - right_img[(i+off+j*size.width)*3+2]);
                }
                unaries(off, i + j* size.width) = diff;
            }
        }
    }

    return unaries;
}

int main(int argc, char *argv[])
{
    if(argc < 2){
        std::cout << "./stereo path_to_stereo_folder" << '\n';
        std::cout << "Example: ./stereo /data/Stereo/tsukuba/" << '\n';
        return 1;
    }

    std::string stereo_folder = argv[1];
    std::string left_image_path = stereo_folder + "imL.png";
    std::string right_image_path = stereo_folder + "imR.png";
    std::string output_image_path = stereo_folder + "out.bmp";

    Potts_weight_set parameters(5, 100, 5, 5, 100);

    img_size size = {-1, -1};

    unsigned char * left_img = load_image(left_image_path, size);
    unsigned char * right_img = load_image(right_image_path, size);


    MatrixXf unaries = get_unaries(left_img, right_img, size);

    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(parameters.spatial_std, parameters.spatial_std,
                            new PottsCompatibility(parameters.spatial_potts_weight));
    crf.addPairwiseBilateral(parameters.bilat_spatial_std, parameters.bilat_spatial_std,
                             parameters.bilat_color_std, parameters.bilat_color_std, parameters.bilat_color_std,
                             left_img, new PottsCompatibility(parameters.bilat_potts_weight));

    clock_t start, end;
    MatrixXf Q = crf.unary_init();
    start = clock();
    Q = crf.qp_inference(Q);
    Q = crf.concave_qp_cccp_inference(Q);
    end = clock();
    double timing = (double(end-start)/CLOCKS_PER_SEC);
    double final_energy = crf.compute_energy(Q);
    double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
    std::cout << "Fractional Energy: " << final_energy << '\n';
    std::cout << "Integer Energy: " << discretized_energy << '\n';

    save_map(Q, size, output_image_path, "Stereo");
}
