#include "file_storage.hpp"
#include "color_to_label.hpp"
#include "probimage.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/stat.h>

// Directory and file stuff
void make_dir(std::string dir_name){
    struct stat resdir_stat;
    if (stat(dir_name.c_str(), &resdir_stat) == -1) {
        mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}


bool file_exist(std::string file_path){
    struct stat path_stat;
    return stat(file_path.c_str(),&path_stat)==0;
}


static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}


void split_string(const std::string &s, const char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> get_all_split_files(const std::string & path_to_dataset, const std::string & split)
{
    std::string path_to_split = path_to_dataset + "split/" + split+ ".txt";

    std::vector<std::string> split_images;
    std::string next_img_name;
    std::ifstream file(path_to_split.c_str());

    while(getline(file, next_img_name)){
        split_images.push_back(rtrim(next_img_name));
    }

    return split_images;
}

std::string stringreplace(std::string s,
                          std::string toReplace,
                          std::string replaceWith)
{
    if (s.find(toReplace) != std::string::npos){
        return(s.replace(s.find(toReplace), toReplace.length(), replaceWith));
    }

    return s;
}


std::string  get_unaries_path(const std::string & path_to_dataset, const std::string & image_name){
    std::string unaries_path = path_to_dataset + "texton_unaries/";
    unaries_path = unaries_path + image_name;
    unaries_path = stringreplace(unaries_path, ".bmp", ".c_unary");
    return unaries_path;
}

std::string  get_image_path(const std::string & path_to_dataset, const std::string & image_name){
    std::string image_path = path_to_dataset +"MSRC_ObjCategImageDatabase_v2/Images/";
    image_path = image_path + image_name;
    return image_path;
}

std::string  get_ground_truth_path(const std::string & path_to_dataset, const std::string & image_name){
    std::string ground_truth_path = path_to_dataset +"MSRC_ObjCategImageDatabase_v2/GroundTruth/";
    ground_truth_path = ground_truth_path + image_name;
    ground_truth_path = stringreplace(ground_truth_path, ".bmp", "_GT.bmp");
    return ground_truth_path;
}

std::string get_output_path(const std::string & path_to_results_folder, const std::string & image_name){
    std::string output_path = path_to_results_folder + image_name;
    output_path = stringreplace(output_path, ".bmp", "_res.bmp");
    return output_path;
}

unsigned char * load_image( const std::string & path_to_image, img_size size){
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

MatrixXf load_unary( const std::string & path_to_unary, img_size& size) {

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

Matrix<short,Dynamic,1> load_labeling(const std::string & path_to_labels, img_size& size){
    Matrix<short,Dynamic,1> labeling(size.width * size.height);

    cv::Mat img = cv::imread(path_to_labels);
    if(size.height != img.rows || size.width != img.cols) {
        std::cout << "Dimension doesn't correspond to labeling" << std::endl;
    }

    labelindex lbl_idx = init_color_to_label_map();

    for (int j=0; j < size.height; j++) {
        for (int i=0; i < size.width; i++) {
            cv::Vec3b intensity = img.at<cv::Vec3b>(j,i); // this comes in BGR
            labeling(j*size.width+i) = lookup_label_index(lbl_idx, intensity);
        }
    }

    return labeling;
}

void save_map(const MatrixXf & estimates, const img_size & size, const std::string & path_to_output) {
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

MatrixXf load_matrix(std::string path_to_matrix){
    std::ifstream infile(path_to_matrix.c_str());

    std::string read;
    int nb_rows, nb_cols;

    std::getline(infile, read, '\t');
    nb_rows = stoi(read);
    std::getline(infile, read);
    nb_cols = stoi(read);

    MatrixXf loaded(nb_rows, nb_cols);

    std::string line;
    std::vector<std::string> all_floats;
    for (int i=0; i < nb_rows; i++) {
        std::getline(infile, line);
        split_string(line, ',', all_floats);
        for (int j=0; j< nb_cols; j++) {
            float new_elt = stof(all_floats[j]);
            loaded(i,j) = new_elt;
        }
        all_floats.resize(0);
    }
    return loaded;
}


void save_matrix(std::string path_to_output, MatrixXf matrix){
    std::ofstream file(path_to_output.c_str());
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    file << matrix.rows() << "\t" << matrix.cols() << std::endl;
    file << matrix.format(CSVFormat);
    file.close();
}
