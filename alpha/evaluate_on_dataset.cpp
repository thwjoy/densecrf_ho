#include <fstream>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "evaluation.hpp"
#include "inference.hpp"
#include "color_to_label.hpp"

#define NUMLABELS 22

/////////////////
// Color Index //
/////////////////
labelindex color_to_label = init_color_to_label_map();


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

void do_inference(std::string path_to_dataset, std::string path_to_results,
                  std::string image_name, std::string path_to_parameters, std::string to_minimize)
{

    std::string image_path = get_image_path(path_to_dataset, image_name);
    std::string unaries_path = get_unaries_path(path_to_dataset, image_name);
    std::string output_path = get_output_path(path_to_results, image_name);

    if(not file_exist(output_path)){
        std::cout << output_path << '\n';
        if (to_minimize == "mf") {
            minimize_mean_field(image_path, unaries_path, output_path, path_to_parameters);
        } else if(to_minimize == "grad"){
            gradually_minimize_mean_field(image_path, unaries_path, output_path, path_to_parameters);
        } else if(to_minimize == "unaries") {
            unaries_baseline(unaries_path, output_path);
        } else{
            float alpha = stof(to_minimize);
            minimize_dense_alpha_divergence(image_path, unaries_path, output_path, path_to_parameters, alpha);
        }
    }
}


/////////////////////////////
// Evalutating the results //
/////////////////////////////
void evaluate_segmentation_files(std::string path_to_dataset, std::string path_to_results, std::string image_name, std::vector<int>& confMat)
{
    std::string gt_path = get_ground_truth_path(path_to_dataset, image_name);
    std::string output_path = get_output_path(path_to_results, image_name);

    cv::Mat gtImg = cv::imread(output_path);
    cv::Mat crfImg = cv::imread(gt_path);

    label_matrix gt_labels = labels_from_lblimg(gtImg, color_to_label);
    label_matrix crf_labels = labels_from_lblimg(crfImg, color_to_label);

    evaluate_segmentation(gt_labels, crf_labels, confMat, NUMLABELS);
}
double compute_pixel_accuracy(std::string dataset_split, std::string path_to_dataset,
                              std::string path_to_generated, std::string path_to_parameters,
                              std::string to_minimize){

    std::vector<std::string> test_images = get_all_split_files(path_to_dataset, dataset_split);

    // Inference
#pragma omp parallel for
    for(int i=0; i< test_images.size(); ++i){
        do_inference(path_to_dataset, path_to_generated, test_images[i], path_to_parameters, to_minimize);
    }

// Confusion evaluation
    std::vector<int> totalConfMat(NUMLABELS * NUMLABELS, 0);
    std::vector<int> conf_mat(NUMLABELS * NUMLABELS, 0);
    std::vector<double> meanIous;
#pragma omp parallel for
    for(int i=0; i < test_images.size(); ++i) {
        std::string image_name = test_images[i];
        std::fill(conf_mat.begin(), conf_mat.end(), 0);
        evaluate_segmentation_files(path_to_dataset, path_to_generated, image_name, conf_mat);

        for(int j = 0; j < NUMLABELS * NUMLABELS; ++j)
        {
            totalConfMat[j] += conf_mat[j];
        }

        meanIous.push_back( compute_mean_iou(conf_mat, NUMLABELS));

    }

    save_confusion_matrix(totalConfMat, path_to_generated + "conf_mat.csv", NUMLABELS);
    save_vector(meanIous, path_to_generated + "mean_iou_per_image.csv");

    return pixwise_acc_from_confmat(totalConfMat, NUMLABELS);
    
}

int main(int argc, char *argv[])
{
    if (argc<3) {
        std::cout << "evaluate split path_to_dataset path_to_results" << '\n';
        std::cout << "Example: ./evaluate Train /home/rudy/datasets/MSRC/ ./train/ learned_parameters.csv -10:-3:-1:2:10:mf:grad" << '\n';
        return 1;
    }

    std::string dataset_split = argv[1];
    std::string path_to_dataset = argv[2];
    std::string path_to_results = argv[3];
    std::string path_to_parameters = argv[4];
    std::string all_alphas = argv[5];

    std::vector<std::string> alphas_to_do;
    split_string(all_alphas, ':', alphas_to_do);

    make_dir(path_to_results);

    for(std::vector<std::string>::iterator alpha_s = alphas_to_do.begin(); alpha_s!= alphas_to_do.end(); ++alpha_s){
        std::string path_to_generated = path_to_results + *alpha_s + '/';
        make_dir(path_to_generated);

        double accuracy = compute_pixel_accuracy(dataset_split, path_to_dataset, path_to_generated, path_to_parameters, *alpha_s);
        std::cout << accuracy << '\n';
        return 0;
    }
}
