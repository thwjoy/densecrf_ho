#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>


#include "inference.hpp"

#define NUMLABELS 21

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


/////////////////
// Color Index //
/////////////////

int lookup_label_index(cv::Vec3b gtVal)
{
    if(gtVal == cv::Vec3b(128,0,0)) return 0;
    else if(gtVal == cv::Vec3b(0,128,0)) return 1;
    else if(gtVal == cv::Vec3b(128,128,0)) return 2;
    else if(gtVal == cv::Vec3b(0,0,128)) return 3;
    else if(gtVal == cv::Vec3b(0,128,128)) return 4;
    else if(gtVal == cv::Vec3b(128,128,128)) return 5;
    else if(gtVal == cv::Vec3b(192,0,0)) return 6;
    else if(gtVal == cv::Vec3b(64,128,0)) return 7;
    else if(gtVal == cv::Vec3b(192,128,0)) return 8;
    else if(gtVal == cv::Vec3b(64,0,128)) return 9;
    else if(gtVal == cv::Vec3b(192,0,128)) return 10;
    else if(gtVal == cv::Vec3b(64,128,128)) return 11;
    else if(gtVal == cv::Vec3b(192,128,128)) return 12;
    else if(gtVal == cv::Vec3b(0,64,0)) return 13;
    else if(gtVal == cv::Vec3b(128,64,0)) return 14;
    else if(gtVal == cv::Vec3b(0,192,0)) return 15;
    else if(gtVal == cv::Vec3b(128,64,128)) return 16;
    else if(gtVal == cv::Vec3b(0,192,128)) return 17;
    else if(gtVal == cv::Vec3b(128,192,128)) return 18;
    else if(gtVal == cv::Vec3b(64,64,0)) return 19;
    else if(gtVal == cv::Vec3b(192,64,0)) return 20;
    else if(gtVal == cv::Vec3b(0,0,0)) return -1;
}


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

double compute_mean_iou (const std::vector<int> & confusionMatrix, int numLabels)
{
    std::set<int> indicesNotConsider;
    indicesNotConsider.insert(-1); // no void labels

    std::vector<int> rowSums;
    std::vector<int> colSums;

    std::vector<double> iouPerClass;
    iouPerClass.clear();

    sum_along_row (confusionMatrix, numLabels, numLabels, rowSums);
    sum_along_col (confusionMatrix, numLabels, numLabels, colSums);

    for (size_t i = 0; i < numLabels; ++i){
        size_t uni = rowSums[i] + colSums[i] - confusionMatrix[numLabels * i + i]; // "union" is a c++ reserved word
        size_t intersection = confusionMatrix[numLabels * i + i];
        iouPerClass.push_back( intersection/ ((double) (uni + std::numeric_limits<double>::epsilon() )) );
    }

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
                  std::string path_to_results, std::string image_name)
{
    std::string image_path = path_to_images + image_name;
    std::string unaries_path = path_to_unaries + image_name;
    unaries_path = stringreplace(unaries_path, ".bmp", ".c_unary");
    std::string output_path = path_to_results + image_name;
    output_path = stringreplace(output_path, ".bmp", "_res.bmp");

    struct stat path_stat;
    if(stat(output_path.c_str(), &path_stat)!=0){
        minimize_dense_alpha_divergence(image_path, unaries_path, output_path,
                                        20, 5, 5);
    }
}


/////////////////////////////
// Evalutating the results //
/////////////////////////////
std::vector<int> evaluate_segmentation(std::string path_to_ground_truths, std::string path_to_results, std::string image_name)
{
    std::string gt_path = path_to_ground_truths + image_name;
    gt_path = stringreplace(gt_path, ".bmp", "_GT.bmp");
    std::string output_path = path_to_results + image_name;
    output_path = stringreplace(output_path, ".bmp", "_res.bmp");

    std::vector<int> confMat(NUMLABELS * NUMLABELS);

    cv::Mat gtImg = cv::imread(gt_path);
    cv::Mat crfImg = cv::imread(output_path);

    assert(gtImg.rows == crfImg.rows);
    assert(gtImg.cols == crfImg.cols);

    for(int y = 0; y < gtImg.rows; ++y)
    {
        for(int x = 0; x < crfImg.rows; ++x)
        {
            cv::Point p(x,y);
            cv::Vec3b gtVal = gtImg.at<cv::Vec3b>(p);
            cv::Vec3b crfVal = crfImg.at<cv::Vec3b>(p);
            std::swap(gtVal[0], gtVal[2]); // since OpenCV uses BGR instead of RGB
            std::swap(crfVal[0], crfVal[2]);
            int gtIndex = lookup_label_index(gtVal);
            int crfIndex = lookup_label_index(crfVal);
            ++confMat[gtIndex * NUMLABELS + crfIndex];
        }
    }

    return confMat;

}


//////////////////////////
// Convenience function //
//////////////////////////
std::vector<std::string> get_all_test_files(std::string path_to_dataset)
{
    std::string path_to_testlist = path_to_dataset + "split/Validation.txt";

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
        std::cout << "evaluate path_to_dataset path_to_results" << '\n';
        return 1;
    }

    std::string path_to_dataset = argv[1];
    std::string path_to_results = argv[2];

    struct stat resdir_stat;
    if (stat(path_to_results.c_str(), &resdir_stat) == -1) {
        mkdir(path_to_results.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }


    std::vector<std::string> test_images = get_all_test_files(path_to_dataset);
    std::string path_to_images = path_to_dataset + "MSRC_ObjCategImageDatabase_v2/Images/";
    std::string path_to_unaries = path_to_dataset + "texton_unaries/";
    std::string path_to_ground_truths = path_to_dataset + "MSRC_ObjCategImageDatabase_v2/GroundTruth/";

    // Inference
    for (std::vector<std::string>::iterator image_name = test_images.begin(); image_name != test_images.end(); ++image_name) {
        do_inference(path_to_images, path_to_unaries, path_to_results, *image_name);
    }


    // Confusion evaluation
    std::vector<int> totalConfMat(NUMLABELS * NUMLABELS);
    std::vector<double> meanIous;
    for(std::vector<std::string>::iterator image_name = test_images.begin(); image_name != test_images.end(); ++image_name) {
        std::vector<int> conf_mat = evaluate_segmentation(path_to_ground_truths, path_to_results, *image_name);

        for(int j = 0; j < NUMLABELS * NUMLABELS; ++j)
        {
            totalConfMat[j] += conf_mat[j];
        }

        meanIous.push_back( compute_mean_iou(conf_mat, NUMLABELS));

    }


    save_confusion_matrix(totalConfMat, path_to_results + "conf_mat.csv", NUMLABELS);
    save_vector(meanIous, path_to_results + "mean_iou_per_image.csv");

    return 0;
}
