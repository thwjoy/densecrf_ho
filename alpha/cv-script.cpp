#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"

void image_inference(Dataset dataset, std::string path_to_results,
                     std::string image_name, float pairweight)
{

    std::string image_path = dataset.get_image_path(image_name);
    std::string unaries_path = dataset.get_unaries_path(image_name);
    std::string dataset_name = dataset.name;

    img_size size;
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(unaries_path, size);
    unsigned char * img = load_image(image_path, size);

    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseBilateral(50, 50,
                             15, 15, 15,
                             img, new PottsCompatibility(pairweight));
    MatrixXf Q;
    // Run Meanfield for 5 iterations
    {
        std::string path_to_subexp_results = path_to_results + "/mf5/";
        std::string output_path = get_output_path(path_to_subexp_results, image_name);
        if (not file_exist(output_path)) {
            Q = crf.inference(5);
            make_dir(path_to_subexp_results);
            double final_energy = crf.compute_energy(Q);
            double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
            save_map(Q, size, output_path, dataset_name);
        }
    }

    // Run Meanfield for 15 iterations
    {
        std::string path_to_subexp_results = path_to_results + "/mf15/";
        std::string output_path = get_output_path(path_to_subexp_results, image_name);
        if (not file_exist(output_path)) {
            Q = crf.inference(15);
            make_dir(path_to_subexp_results);
            double final_energy = crf.compute_energy(Q);
            double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
            save_map(Q, size, output_path, dataset_name);
        }
    }
    // Run Meanfield for 25 iterations
    {
        std::string path_to_subexp_results = path_to_results + "/mf25/";
        std::string output_path = get_output_path(path_to_subexp_results, image_name);
        if (not file_exist(output_path)) {
            Q = crf.inference(25);
            make_dir(path_to_subexp_results);
            double final_energy = crf.compute_energy(Q);
            double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
            save_map(Q, size, output_path, dataset_name);
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc<4) {
        std::cout << "./generate split dataset results params" << '\n';
        std::cout << "Example: ./generate Train MSRC /data/MSRC/results/train/ 3" << '\n';
        return 1;
    }

    std::string dataset_split = argv[1];
    std::string dataset_name  = argv[2];
    std::string path_to_results = argv[3];

    std::string param1 = argv[4];
    float pair_weight = std::stof(param1);

    make_dir(path_to_results);

    Dataset ds = get_dataset_by_name(dataset_name);
    std::vector<std::string> test_images = ds.get_all_split_files(dataset_split);
    omp_set_num_threads(8);
#pragma omp parallel for
    for(int i=0; i< test_images.size(); ++i){
        image_inference(ds, path_to_results, test_images[i], pair_weight);
    }


}
