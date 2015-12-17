#include "inference.hpp"
#include "evaluation.hpp"
#include "file_storage.hpp"


int main(int argc, char *argv[])
{
    std::string path_to_dataset = "/home/rudy/datasets/MSRC/";
    std::string dataset_split = "Validation";

    std::vector<std::string> test_images = get_all_split_files(path_to_dataset, dataset_split);

    for (float w1 =5; w1< 11; w1++) {
        for (float sigma_alpha = 50; sigma_alpha<110; sigma_alpha+=10) {
            for (float sigma_beta=3; sigma_beta<11; sigma_beta++) {

#pragma omp parallel for
                for(int i=0; i< test_images.size(); ++i){
                    std::string image_name = test_images[i];
                    std::string image_path = get_image_path(path_to_dataset, image_name);
                    std::string unaries_path = get_unaries_path(path_to_dataset, image_name);
                    std::string gt_path = get_ground_truth_path(path_to_dataset, image_name);

                    label_matrix gt = load_label_matrix(gt_path);
                    label_matrix img_res = minimize_mean_field(image_path, unaries_path, w1, sigma_alpha, sigma_beta);
                }
            }
        }
    }

    return 0;
}
