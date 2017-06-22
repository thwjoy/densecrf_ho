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
                     std::string image_name, float spc_std, float spc_potts, float bil_spcstd, float bil_colstd, float bil_potts, float sp_const, sp_params params)
{

    std::string image_path = dataset.get_image_path(image_name);
    std::string unaries_path = dataset.get_unaries_path(image_name);
    std::string dataset_name = dataset.name;
    std::string super_pixel_path = "./data/MSRC/MSRC_ObjCategImageDatabase_v2/SuperPixels";

    img_size size = {-1, -1};
    //Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(unaries_path, size);
    unsigned char * img = load_image(image_path, size);

    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,bil_colstd, bil_colstd, bil_colstd,img, new PottsCompatibility(bil_potts));

    MatrixXf Q;
    {
        std::string path_to_subexp_results = path_to_results + "/" + method + "/";
        std::string output_path = get_output_path(path_to_subexp_results, image_name);

       if (not file_exist(output_path)) {
            clock_t start, end;
            double timing;
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
            } else if (method == "qp_nc"){ //qp with a non convex energy function, relaxations removed
                //std::cout << "---Running tests on QP with non convex energy\r\n";
                //Q = crf.qp_inference(Q);
                crf.addSuperPixel(super_pixel_path + "/400/" + image_name + "_clsfr.bin",img,params.const_1,params.norm_1);
                crf.addSuperPixel(super_pixel_path + "/100/" + image_name + "_clsfr.bin",img,params.const_2,params.norm_2); 
                crf.addSuperPixel(super_pixel_path + "/250/" + image_name + "_clsfr.bin",img,params.const_3,params.norm_3); 
                Q = crf.qp_inference_non_convex(Q);
            } else if (method == "qp_sp_0" || method == "qp_sp_0" || method == "qp_sp_00001" || method == "qp_sp_0001" || method == "qp_sp_001" || method == "qp_sp_01" || method == "qp_sp_1" || method == "qp_sp_10" || method == "qp_sp_100" || method == "qp_sp_1000" || method == "qp_sp_10000" ){
                //std::cout << "---Running tests on QP with super pixel terms, with constant = "<< sp_const << "\r\n";
                crf.addSuperPixel(super_pixel_path + "/400/" + image_name + "/_clsfr.bin",img,params.const_1,params.norm_1);
                crf.addSuperPixel(super_pixel_path + "/100/" + image_name + "/_clsfr.bin",img,params.const_2,params.norm_2); 
                crf.addSuperPixel(super_pixel_path + "/40/" + image_name + "/_clsfr.bin",img,params.const_3,params.norm_3);             
                Q = crf.qp_inference_super_pixels_non_convex(Q);
            } else if (method == "qp_sp"){ //qp with a non convex energy function, relaxations removed
                //std::cout << "---Running tests on QP with non convex energy and Super Pixels\r\n";
                crf.addSuperPixel(super_pixel_path + "/400/" + image_name + "_clsfr.bin",img,params.const_1,params.norm_1);
                crf.addSuperPixel(super_pixel_path + "/100/" + image_name + "_clsfr.bin",img,params.const_2,params.norm_2); 
                crf.addSuperPixel(super_pixel_path + "/250/" + image_name + "_clsfr.bin",img,params.const_3,params.norm_3);              
                Q = crf.qp_inference_super_pixels_non_convex(Q);
            } else if (method == "lrqp_sp"){ //qp with a non convex energy function, relaxations removed
                //std::cout << "---Running tests on QP with non convex energy and Super Pixels\r\n";
                crf.addSuperPixel(super_pixel_path + "/400/" + image_name + "_clsfr.bin",img,params.const_1,params.norm_1);
                crf.addSuperPixel(super_pixel_path + "/100/" + image_name + "_clsfr.bin",img,params.const_2,params.norm_2); 
                crf.addSuperPixel(super_pixel_path + "/250/" + image_name + "_clsfr.bin",img,params.const_3,params.norm_3);              
                Q = crf.qp_inference_super_pixels(Q);
            } else if (method == "unary"){
                (void)0;
            } else{
                //std::cout << "Unrecognised method.\n Proper error handling would do something but I won't." << '\n';
            }


            make_dir(path_to_subexp_results);
            end = clock();
            timing = (double(end-start)/CLOCKS_PER_SEC);
            double final_energy = crf.compute_energy(Q);
            double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
            save_map(Q, size, output_path, dataset_name);
            std::string txt_output = output_path;
            txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
            std::ofstream txt_file(txt_output.c_str());
            txt_file << timing << '\t' << final_energy << '\t' << discretized_energy << std::endl;
            txt_file.close();
        } else {
            //std::cout << "File exists! Skipping: " << image_name << std::endl; 
        }
    }
    delete[] img; 
}

int main(int argc, char *argv[])
{
    if (argc<4) {
        std::cout << "./generate split dataset method results_path spc_std spc_potts bil_spcstd bil_colstd bil_potts" << '\n';
        std::cout << "Example: ./generate Validation Pascal2010 method /data/MSRC/results/train/ 3 38 40 5 50" << '\n';
        return 1;
    }


    std::string dataset_split = argv[1];
    std::string dataset_name  = argv[2];
    std::string method = argv[3];
    std::string path_to_results = argv[4];
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
    if(method == "qp_sp_0") sp_const = 0;
    else if (method == "qp_sp_1") sp_const = 1;
    else if (method == "qp_sp_10") sp_const = 10;
    else if (method == "qp_sp_100") sp_const = 100;
    else if (method == "qp_sp_1000") sp_const = 1000;
    else if (method == "qp_sp_10000") sp_const = 10000;
    else if (method == "qp_sp_01") sp_const = 0.1;
    else if (method == "qp_sp_001") sp_const = 0.01;
    else if (method == "qp_sp_0001") sp_const = 0.001;
    else if (method == "qp_sp_00001") sp_const = 0.0001;
  
    make_dir(path_to_results);

    Dataset ds = get_dataset_by_name(dataset_name);
    std::vector<std::string> test_images;
    if (dataset_name == "MSRC") test_images = ds.get_MSRC_split_files(dataset_split);
    else test_images = ds.get_all_split_files(dataset_split);
//    omp_set_num_threads(4);
//#pragma omp parallel for
    for(int i=0; i< test_images.size(); ++i){
        //std::cout << "Image: " << test_images[i] << "\t" << i << "/" << test_images.size() << std::endl;
        image_inference(ds, method, path_to_results,  test_images[i], spc_std, spc_potts, bil_spcstd, bil_colstd, bil_potts, sp_const, params);
    }


}
