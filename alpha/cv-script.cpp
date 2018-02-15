#include <chrono>
#include <ctime>
#include <fstream>
#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"


struct sp_params {
    float const_1;
    float const_2;
    float const_3;
    float norm_1;
    float norm_2;
    float norm_3;
};


void image_inference(Dataset dataset, std::string method, std::string path_to_results,
                     std::string image_name, float spc_std, float spc_potts, 
                     float bil_spcstd, float bil_colstd, float bil_potts, LP_inf_params & lp_params, double sp_const, sp_params params)
{

    std::string image_path = dataset.get_image_path(image_name);
    std::string unaries_path = dataset.get_unaries_path(image_name);
    std::string dataset_name = dataset.name;
    std::string super_pixel_path = "/media/tom/DATA/datasets/MSRC/SuperPixels";
    std::string output_path = get_output_path(path_to_results, image_name);

    MatrixXf Q;
    {
        if (not file_exist(output_path)) {


            
    	    img_size size = {-1, -1};
    	    //Load the unaries potentials for our image.
    	    MatrixXf unaries = load_unary(unaries_path, size);
    	    unsigned char * img = load_image(image_path, size);

    	    DenseCRF2D crf(size.width, size.height, unaries.rows());
    	    crf.setUnaryEnergy(unaries);
    	    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    	    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,bil_colstd, bil_colstd, bil_colstd,img, new PottsCompatibility(bil_potts));
            typedef std::chrono::high_resolution_clock::time_point htime;
            clock_t start, end;
            //time_t start, end;
            //double start, end;
            double timing;
            //std::cout << image_path << std::endl;
            //start = clock();
            std::vector<perf_measure> traced_perfs;
            std::vector<perf_measure> new_perfs;
            std::vector<int> pixel_ids;
            double time_budget = 30;    // seconds

            start = clock();
            Q = crf.unary_init();
            if (method == "prox_lp_sp"){    // standard prox_lp
                crf.addSuperPixel(super_pixel_path + "/400/" + image_name + "_clsfr.bin",img,params.const_1,params.norm_1);
                crf.addSuperPixel(super_pixel_path + "/100/" + image_name + "_clsfr.bin",img,params.const_2,params.norm_2); 
                crf.addSuperPixel(super_pixel_path + "/250/" + image_name + "_clsfr.bin",img,params.const_3,params.norm_3); 
                try {
                    Q = crf.lp_inference_prox_super_pixels(Q, lp_params);   
                } catch (std::runtime_error &e) {
                    std::cout << "Runtime Error!: " << e.what() << std::endl;
                }
            } else if (method == "prox_lp") {
                Q = crf.lp_inference_prox(Q, lp_params);
            } else if (method == "qp_nc"){ //qp with a non convex energy function, relaxations removed
                //std::cout << "---Running tests on QP with non convex energy\r\n";
                Q = crf.qp_inference_non_convex(Q);
            } else if (method == "qp_sp"){ //qp with a non convex energy function, relaxations removed
                //std::cout << "---Running tests on QP with non convex energy and Super Pixels\r\n";
                crf.addSuperPixel(super_pixel_path + "/400/" + image_name + "_clsfr.bin",img,params.const_1,params.norm_1);
                crf.addSuperPixel(super_pixel_path + "/100/" + image_name + "_clsfr.bin",img,params.const_2,params.norm_2); 
                crf.addSuperPixel(super_pixel_path + "/250/" + image_name + "_clsfr.bin",img,params.const_3,params.norm_3);              
                Q = crf.qp_inference_super_pixels_non_convex(Q);
            } else if (method == "unary"){
                (void)0;
            } else{
                std::cout << "Unrecognised method.\n Proper error handling would do something but I won't." << '\n';
            }


            make_dir(path_to_results);
            end = clock();
            timing = (double(end-start)/CLOCKS_PER_SEC);
            double final_energy, discretized_energy;
            if (method == "qp_sp")
            {
                final_energy = crf.compute_energy_higher_order(Q);
                discretized_energy = crf.assignment_energy_higher_order(crf.currentMap(Q));
            }
            else if (method == "prox_lp_sp")
            {
                final_energy = crf.compute_energy_LP_higher_order(Q);
                discretized_energy = crf.assignment_energy_higher_order(crf.currentMap(Q));
            }
            {
                final_energy = crf.compute_energy(Q);
                discretized_energy = crf.assignment_energy(crf.currentMap(Q));
            }
            save_map(Q, size, output_path, dataset_name);
            if (!pixel_ids.empty()) save_less_confident_pixels(Q, pixel_ids, size, output_path, dataset_name);
            std::string txt_output = output_path;
            txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
            std::ofstream txt_file(txt_output);
            txt_file << timing << '\t' << final_energy << '\t' << discretized_energy << std::endl;
            //std::cout << "#" << method << ": " << timing << '\t' << final_energy << '\t' << discretized_energy << std::endl;
            txt_file.close();
            delete[] img;
        } else {
            std::cout << "File exists! Skipping: " << image_name << std::endl;
        }
    }

}

int main(int argc, char *argv[])
{
    if (argc<4) {
        std::cout << "./generate split dataset method results_path spc_std spc_potts bil_spcstd bil_colstd bil_potts lp_params[7]" << '\n';
        std::cout << "Example: ./generate Validation Pascal2010 method /data/MSRC/results/train/ 3 38 40 5 50" << '\n';
        return 1;
    }


    std::string dataset_split = argv[1];
    std::string dataset_name  = argv[2];
    std::string method = argv[3];
    std::string path_to_results = argv[4];
    make_dir(path_to_results);
    path_to_results += std::string("/") + method;
    make_dir(path_to_results);
    path_to_results += std::string("/") + dataset_split + std::string("/");
    make_dir(path_to_results);
    //path_to_results = "/home/tomj/Documents/4YP/densecrf/" + path_to_results;
    //std::cout << "build/alpha/cv-script " << dataset_split << " " << dataset_name << " " << method << " " << path_to_results << " " << argv[5] << " " << argv[6]  << " " << argv[7]  << " " << argv[8]  << " " << argv[9]  << std::endl;


    std::string param1 = argv[5];
    float spc_std = std::stof(param1);
    std::string param2 = argv[6];
    float spc_potts = std::stof(param2);
    std::string param3 = argv[7];
    float bil_spcstd = std::stof(param3);
    std::string param4 = argv[8];
    float bil_colstd = std::stof(param4);
    std::string param5 = argv[9];
    float bil_potts = std::stof(param5);
    sp_params params;
    if (argc==16) params = sp_params {std::stof(argv[10]), std::stof(argv[11]), std::stof(argv[12]), std::stof(argv[13]), std::stof(argv[14]), std::stof(argv[15])};
    else params = {0,0,0,0,0,0};


    float sp_const;

    // lp inference params
    LP_inf_params lp_params;
    lp_params.prox_energy_tol = lp_params.dual_gap_tol;

  
    make_dir(path_to_results);

    std::cout << "## COMMAND: " << argv[0] << " " << dataset_split << " " << dataset_name << " " << method << " "
        << path_to_results << " " << spc_std << " " << spc_potts << " " << bil_spcstd << " " << bil_colstd << " "
        << bil_potts << " "
        << lp_params.prox_max_iter << " " << lp_params.fw_max_iter << " " << lp_params.qp_max_iter << " "
        << lp_params.prox_reg_const << " " << lp_params.dual_gap_tol << " " << lp_params.qp_tol << " "
        << lp_params.best_int << " " << lp_params.prox_energy_tol << std::endl;

    Dataset ds = get_dataset_by_name(dataset_name, std::string("/media/tom/DATA/datasets/MSRC"));
    std::vector<std::string> test_images;
    if (dataset_name == "MSRC") test_images = ds.get_MSRC_split_files(dataset_split);
    else test_images = ds.get_all_split_files(dataset_split);
    //#pragma omp parallel for num_threads(12)
    for(int i=0; i< test_images.size(); ++i){
        image_inference(ds, method, path_to_results,  test_images[i], spc_std, spc_potts,
                        bil_spcstd, bil_colstd, bil_potts, lp_params, sp_const, params);
    }



}
