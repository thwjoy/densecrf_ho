#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"
#include <iostream>

#define NUM_LABELS 16

MatrixXf get_unaries(unsigned char * left_img, unsigned char * right_img, img_size & size){
    MatrixXf unaries(NUM_LABELS, size.height * size.width);

    for (int off=0; off<NUM_LABELS; off++) {
        for (int j = 0; j < size.height; j++) {
            for (int i=0; i < size.width; i++) {
                float diff = 0;
                if (i + off < size.width) {
                    diff += abs(left_img[(i+j*size.width)*3+0] - right_img[(i+off+j*size.width)*3+0]);
                    diff += abs(left_img[(i+j*size.width)*3+1] - right_img[(i+off+j*size.width)*3+1]);
                    diff += abs(left_img[(i+j*size.width)*3+2] - right_img[(i+off+j*size.width)*3+2]);
                }
                unaries(off, diff);
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
    }

    std::string stereo_folder = argv[1];
    std::string left_image_path = stereo_folder + "imL.png";
    std::string right_image_path = stereo_folder + "imR.png";

    img_size size = {-1, 1};

    unsigned char * left_img = load_image(left_image_path, size);
    unsigned char * right_img = load_image(right_image_path, size);


    MatrixXf unaries = get_unaries(left_img, right_img, size);





}
