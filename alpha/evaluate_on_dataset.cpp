#include <omp.h>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <limits>
#include <sstream>

#include "inference.hpp"
#include "color_to_label.hpp"

#define NUMLABELS 22

///////////////////////////////////
// String Manipulation Functions //
///////////////////////////////////

static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
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

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
void make_dir(std::string dir_name){
    struct stat resdir_stat;
    if (stat(dir_name.c_str(), &resdir_stat) == -1) {
        mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

/////////////////
// Color Index //
/////////////////
labelindex color_to_label = init_color_to_label_map();

///////////////////////////////////
// Confusion Matrix manipulation //
///////////////////////////////////

void sum_along_row (const std::vector<int> & matrix, int n_rows, int n_cols, std::vector<int>& sums)
{
    sums.clear();
    sums.assign(n_rows, 0);

    for (size_t row = 0; row < n_rows; ++row){
        for (size_t col = 0; col < n_cols; ++col){
            sums[row] += matrix[row*n_rows + col];
        }
    }
}

void sum_along_col (const std::vector<int> & matrix, int n_rows, int n_cols, std::vector<int>& sums)
{
    sums.clear();
    sums.assign(n_cols, 0);

    for (size_t col = 0; col < n_cols; ++col){
        for (size_t row = 0; row < n_rows; ++row){
            sums[col] += matrix[row*n_rows + col];
        }
    }
}

template <typename T>
double mean_vector (const std::vector<T> & vector, std::set<int> & indicesNotConsider)
{
    double mean = 0;
    double N = 0;

    for (size_t i = 0; i < vector.size(); ++i){
        if ( indicesNotConsider.find(i) == indicesNotConsider.end() ){
            mean = mean + ( (double) vector[i] );
            ++N;
        }
    }

    return mean/N;
}


void find_blank_gt (const std::vector<int> & rowSums, std::set<int> & indicesNotConsider)
{
    for (int i = 0; i < rowSums.size(); ++i){
        if (rowSums[i] == 0) {
            indicesNotConsider.insert(i);
        }
    }
}


double compute_mean_iou (const std::vector<int> & confusionMatrix, int numLabels)
{
    std::set<int> indicesNotConsider;
    indicesNotConsider.insert(21); // no void labels

    std::vector<int> rowSums;
    std::vector<int> colSums;

    std::vector<double> iouPerClass;
    iouPerClass.clear();

    sum_along_row (confusionMatrix, numLabels, numLabels, rowSums);
    sum_along_col (confusionMatrix, numLabels, numLabels, colSums);

    for (size_t i = 0; i < numLabels; ++i){
        size_t uni = rowSums[i] + colSums[i] - confusionMatrix[numLabels * i + i]; // "union" is a c++ reserved word
        size_t intersection = confusionMatrix[numLabels * i + i];
        double iou = intersection/ ((double) (uni + std::numeric_limits<double>::epsilon() ));

        iouPerClass.push_back(iou);
    }
    // Necessary because in the case of a label not present in a GT,
    // this would mean an IoU of zero which is exagerated
    find_blank_gt (rowSums, indicesNotConsider);
    double mean = mean_vector(iouPerClass, indicesNotConsider);

    return mean;
}


////////////////////////////////////////////////////////
// Save the results for analysis of the Segmentation  //
////////////////////////////////////////////////////////

void save_confusion_matrix(const std::vector<int>& confMat, const std::string& filename, const size_t num_labels)
{
  std::ofstream fs(filename.c_str());
  for(int i = 0; i < num_labels * num_labels; ++i)
  {
    fs << confMat[i] << (i % num_labels == (num_labels-1) ? '\n' : ',');
  }
}

template <typename T>
void save_vector(const std::vector<T>& vector, const std::string& filename)
{

  const size_t num_elements = vector.size();

  std::ofstream fs(filename.c_str());
  for(int i = 0; i < num_elements; ++i)
  {
    fs << vector[i] << (i % num_elements == (num_elements-1) ? '\n' : ',');
  }
}

//////////////////////////////
// Performing the inference //
//////////////////////////////

void do_inference(std::string path_to_images, std::string path_to_unaries,
                  std::string path_to_results, std::string image_name,
                  std::string to_minimize)
{
    std::string image_path = path_to_images + image_name;
    std::string unaries_path = path_to_unaries + image_name;
    unaries_path = stringreplace(unaries_path, ".bmp", ".c_unary");
    std::string output_path = path_to_results + image_name;
    output_path = stringreplace(output_path, ".bmp", "_res.bmp");

    struct stat path_stat;
    if(stat(output_path.c_str(), &path_stat)!=0){
        std::cout << output_path << '\n';
        if (to_minimize == "mf") {
            minimize_mean_field(image_path, unaries_path, output_path);
        } else if(to_minimize == "grad"){
            gradually_minimize_mean_field(image_path, unaries_path, output_path);
        } else if(to_minimize == "unaries") {
            unaries_baseline(unaries_path, output_path);
        } else{
            float alpha = stof(to_minimize);
            minimize_dense_alpha_divergence(image_path, unaries_path, output_path, alpha);
        }
    }
}


/////////////////////////////
// Evalutating the results //
/////////////////////////////
void evaluate_segmentation(std::string path_to_ground_truths, std::string path_to_results, std::string image_name, std::vector<int>& confMat)
{
    std::string gt_path = path_to_ground_truths + image_name;
    gt_path = stringreplace(gt_path, ".bmp", "_GT.bmp");
    std::string output_path = path_to_results + image_name;
    output_path = stringreplace(output_path, ".bmp", "_res.bmp");

    cv::Mat gtImg = cv::imread(output_path);
    cv::Mat crfImg = cv::imread(gt_path);

    assert(gtImg.rows == crfImg.rows);
    assert(gtImg.cols == crfImg.cols);

    for(int y = 0; y < gtImg.rows; ++y)
    {
        for(int x = 0; x < gtImg.cols; ++x)
        {
            cv::Point p(x,y);
            cv::Vec3b gtVal = gtImg.at<cv::Vec3b>(p);
            cv::Vec3b crfVal = crfImg.at<cv::Vec3b>(p);
            std::swap(gtVal[0], gtVal[2]); // since OpenCV uses BGR instead of RGB
            std::swap(crfVal[0], crfVal[2]);
            int gtIndex = lookup_label_index(color_to_label, gtVal);
            int crfIndex = lookup_label_index(color_to_label, crfVal);
            ++confMat[gtIndex * NUMLABELS + crfIndex];
        }
    }

}


//////////////////////////
// Convenience function //
//////////////////////////
std::vector<std::string> get_all_test_files(std::string path_to_dataset, std::string split)
{
    std::string path_to_testlist = path_to_dataset + "split/" + split+ ".txt";

    std::vector<std::string> test_images;
    std::string next_img_name;
    std::ifstream file(path_to_testlist.c_str());

    while(getline(file, next_img_name)){
        test_images.push_back(rtrim(next_img_name));
    }

    return test_images;
}



int main(int argc, char *argv[])
{
    if (argc<3) {
        std::cout << "evaluate split path_to_dataset path_to_results" << '\n';
        std::cout << "Example: ./evaluate Train /home/rudy/datasets/MSRC/ ./train/ -10:-3:-1:2:10:mf:grad" << '\n';
        return 1;
    }

    std::string dataset_split = argv[1];
    std::string path_to_dataset = argv[2];
    std::string path_to_results = argv[3];
    std::string all_alphas = argv[4];

    std::vector<std::string> test_images = get_all_test_files(path_to_dataset, dataset_split);
    std::string path_to_images = path_to_dataset + "MSRC_ObjCategImageDatabase_v2/Images/";
    std::string path_to_unaries = path_to_dataset + "texton_unaries/";
    std::string path_to_ground_truths = path_to_dataset + "MSRC_ObjCategImageDatabase_v2/GroundTruth/";


    std::vector<std::string> alphas_to_do;
    split(all_alphas, ':', alphas_to_do);

    make_dir(path_to_results);

    for(std::vector<std::string>::iterator alpha_s = alphas_to_do.begin(); alpha_s!= alphas_to_do.end(); alpha_s++){
        std::string path_to_generated = path_to_results + *alpha_s + '/';

        make_dir(path_to_generated);

        // Inference
#pragma omp parallel for
        for(int i=0; i< test_images.size(); ++i){
            do_inference(path_to_images, path_to_unaries, path_to_generated, test_images[i], *alpha_s);
        }

        // Confusion evaluation
        std::vector<int> totalConfMat(NUMLABELS * NUMLABELS, 0);
        std::vector<int> conf_mat(NUMLABELS * NUMLABELS, 0);
        std::vector<double> meanIous;
        for(std::vector<std::string>::iterator image_name = test_images.begin(); image_name != test_images.end(); ++image_name) {
            std::fill(conf_mat.begin(), conf_mat.end(), 0);
            evaluate_segmentation(path_to_ground_truths, path_to_generated, *image_name, conf_mat);

            for(int j = 0; j < NUMLABELS * NUMLABELS; ++j)
            {
                totalConfMat[j] += conf_mat[j];
            }

            meanIous.push_back( compute_mean_iou(conf_mat, NUMLABELS));

        }


        save_confusion_matrix(totalConfMat, path_to_generated + "conf_mat.csv", NUMLABELS);
        save_vector(meanIous, path_to_generated + "mean_iou_per_image.csv");

    }
    return 0;
}
