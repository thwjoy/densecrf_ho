#include <fstream>
#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"

void image_inference(Dataset dataset, std::string path_to_results,
                     std::string image_name, float spc_std, float spc_potts, float bil_spcstd, float bil_colstd, float bil_potts)
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
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new PottsCompatibility(bil_potts));
    MatrixXf Q;
    {
        std::string path_to_subexp_results = path_to_results + "/lrqp/";
        std::string output_path = get_output_path(path_to_subexp_results, image_name);
        if (not file_exist(output_path)) {
            Q = crf.qp_inference();
            make_dir(path_to_subexp_results);
            double final_energy = crf.compute_energy(Q);
            double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
            save_map(Q, size, output_path, dataset_name);
            std::string txt_output = output_path;
            txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
            std::ofstream txt_file(txt_output.c_str());
            txt_file << 0 << '\t' << final_energy << '\t' << discretized_energy << std::endl;
            txt_file.close();
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
        float spc_std = std::stof(param1);
        std::string param2 = argv[5];
        float spc_potts = std::stof(param2);
        std::string param3 = argv[6];
        float bil_spcstd = std::stof(param3);
        std::string param4 = argv[7];
        float bil_colstd = std::stof(param4);
        std::string param5 = argv[8];
        float bil_potts = std::stof(param5);

        make_dir(path_to_results);

        Dataset ds = get_dataset_by_name(dataset_name);
        std::vector<std::string> test_images = ds.get_all_split_files(dataset_split);
        omp_set_num_threads(8);
#pragma omp parallel for
        for(int i=0; i< test_images.size(); ++i){
            image_inference(ds, path_to_results, test_images[i], spc_std, spc_potts,
                            bil_spcstd, bil_colstd, bil_potts);
        }


    }