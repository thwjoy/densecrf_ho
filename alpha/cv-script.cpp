#include <fstream>
#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"

void image_inference(Dataset dataset, std::string method, std::string path_to_results,
                     std::string image_name, float spc_std, float spc_potts, 
                     float bil_spcstd, float bil_colstd, float bil_potts, LP_inf_params & lp_params)
{

    std::string image_path = dataset.get_image_path(image_name);
    std::string unaries_path = dataset.get_unaries_path(image_name);
    std::string dataset_name = dataset.name;

    img_size size = {-1, -1};
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
        std::string path_to_subexp_results = path_to_results + "/" + method + "/";
        std::string output_path = get_output_path(path_to_subexp_results, image_name);
        if (not file_exist(output_path)) {
            clock_t start, end;
            double timing;
            std::cout << image_path << std::endl;
            start = clock();
            Q = crf.unary_init();
            if (method == "mf5") {
                Q = crf.inference(Q, 5);
            } else if (method == "mf") {
                Q = crf.inference(Q);
            } else if (method == "lrqp") {
                Q = crf.qp_inference(Q);
            } else if (method == "qpcccp") {
                Q = crf.qp_inference(Q);
                Q = crf.qp_cccp_inference(Q);
            } else if (method == "fixedDC-CCV"){
                Q = crf.qp_inference(Q);
                Q = crf.concave_qp_cccp_inference(Q);
            } else if (method == "sg_lp"){
                Q = crf.qp_inference(Q);
                Q = crf.concave_qp_cccp_inference(Q);
                Q = crf.lp_inference(Q, false);
            } else if (method == "cg_lp"){
                Q = crf.qp_inference(Q);
                Q = crf.concave_qp_cccp_inference(Q);
                Q = crf.lp_inference(Q, true);
            } else if (method == "prox_lp"){
                Q = crf.qp_inference(Q);
                Q = crf.concave_qp_cccp_inference(Q);
                Q = crf.lp_inference_prox(Q, lp_params);
            } else if (method == "unary"){
                (void)0;
            } else{
                std::cout << "Unrecognised method.\n Proper error handling would do something but I won't." << '\n';
            }


            make_dir(path_to_subexp_results);
            end = clock();
            timing = (double(end-start)/CLOCKS_PER_SEC);
            double final_energy = crf.compute_energy_true(Q);
            double discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
            save_map(Q, size, output_path, dataset_name);
            std::string txt_output = output_path;
            txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
            std::ofstream txt_file(txt_output.c_str());
            txt_file << timing << '\t' << final_energy << '\t' << discretized_energy << std::endl;
            txt_file.close();
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

    // lp inference params
	LP_inf_params lp_params;
	if(argc > 10) lp_params.prox_max_iter = atoi(argv[10]);
	if(argc > 11) lp_params.fw_max_iter = atoi(argv[11]);
	if(argc > 12) lp_params.qp_max_iter = atoi(argv[12]);
	if(argc > 13) lp_params.prox_reg_const = atof(argv[13]);
	if(argc > 14) lp_params.dual_gap_tol = atof(argv[14]);
	if(argc > 15) lp_params.qp_tol = atof(argv[15]);
	if(argc > 16) lp_params.best_int = atoi(argv[16]);
    lp_params.prox_energy_tol = lp_params.dual_gap_tol;
	if(argc > 17) lp_params.prox_energy_tol = atof(argv[17]);

    std::cout << "## COMMAND: " << argv[0] << " " << dataset_split << " " << dataset_name << " " << method << " "
        << path_to_results << " " << spc_std << " " << spc_potts << " " << bil_spcstd << " " << bil_colstd << " "
        << bil_potts << " " 
        << lp_params.prox_max_iter << " " << lp_params.fw_max_iter << " " << lp_params.qp_max_iter << " "
        << lp_params.prox_reg_const << " " << lp_params.dual_gap_tol << " " << lp_params.qp_tol << " " 
        << lp_params.best_int << " " << lp_params.prox_energy_tol << std::endl;

    make_dir(path_to_results);

    Dataset ds = get_dataset_by_name(dataset_name);
    std::vector<std::string> test_images = ds.get_all_split_files(dataset_split);
    omp_set_num_threads(2);
#pragma omp parallel for
    for(int i=0; i< test_images.size(); ++i){
        image_inference(ds, method, path_to_results,  test_images[i], spc_std, spc_potts,
                        bil_spcstd, bil_colstd, bil_potts, lp_params);
    }


}
