#include "file_storage.hpp"
#include "probimage.h"
#include <iostream>
#include <opencv2/opencv.hpp>


unsigned char * load_image( const std::string path_to_image, img_size size){
    cv::Mat img = cv::imread(path_to_image);

    if(size.height != img.rows || size.width != img.cols) {
        std::cout << "Dimension doesn't correspond to unaries" << std::endl;
    }

    unsigned char * char_img = new unsigned char[size.width*size.height*3];
    for (int j=0; j < size.height; j++) {
        for (int i=0; i < size.width; i++) {
            cv::Vec3b intensity = img.at<cv::Vec3b>(j,i); // this comes in BGR
            char_img[(i+j*size.width)*3+0] = intensity.val[2];
            char_img[(i+j*size.width)*3+1] = intensity.val[1];
            char_img[(i+j*size.width)*3+2] = intensity.val[0];
        }
    }

    return char_img;
}

MatrixXf load_unary( const std::string path_to_unary, img_size& size) {

    ProbImage texton;
    texton.decompress(path_to_unary.c_str());
    texton.boostToProb();

    MatrixXf unaries( texton.depth(), texton.width() * texton.height());
    int i,j,k;
    for(i=0; i<texton.height(); ++i){
        for(j=0; j<texton.width(); ++j){
            for(k=0; k<texton.depth(); ++k){
                // careful with the index position, the operator takes
                // x (along width), then y (along height)

                // Also note that these are probabilities, what we
                // want are unaries, so we need to
                unaries(k, i*texton.width() + j) = -log( texton(j,i,k));
            }
        }
    }

    size = {texton.width(), texton.height()};

    return unaries;
}

VectorXi read_labeling(const std::string path_to_labels, img_size& size){
    VectorXi labeling(size.width * size.height);

    cv::Mat img = cv::imread(path_to_labels);
    if(size.height != img.rows || size.width != img.cols) {
        std::cout << "Dimension doesn't correspond to labeling" << std::endl;
    }

    for (int j=0; j < size.height; j++) {
        for (int i=0; i < size.width; i++) {
            cv::Vec3b intensity = img.at<cv::Vec3b>(j,i); // this comes in BGR

        }
    }



}

void save_map(const MatrixXf estimates, const img_size size, const std::string path_to_output) {
    std::vector<short> labeling(estimates.cols());

    // MAP estimation
    for(int i=0; i<estimates.cols(); ++i) {
        int lbl;
        estimates.col(i).maxCoeff( &lbl);
        labeling[i] = lbl;
    }

    // Make the image
    cv::Mat img(size.height, size.width, CV_8UC3);
    cv::Vec3b intensity;
    for(int i=0; i<estimates.cols(); ++i) {
        intensity[2] = legend[3*labeling[i]];
        intensity[1] = legend[3*labeling[i] + 1];
        intensity[0] = legend[3*labeling[i] + 2];

        int col = i % size.width;
        int row = (i - col)/size.width;
        img.at<cv::Vec3b>(row, col) = intensity;
    }

    cv::imwrite(path_to_output, img);
}
