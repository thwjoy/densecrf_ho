#include "file_storage.hpp"

class Potts_weight_set {
public:
    float spatial_std, spatial_potts_weight,
        bilat_spatial_std, bilat_color_std, bilat_potts_weight;
    Potts_weight_set(float spatial_std, float spatial_potts_weight,
                     float bilat_spatial_std, float bilat_color_std, float bilat_potts_weight);
};


void minimize_mean_field_fixed_iter(std::string path_to_image, std::string path_to_unaries,
                                    Potts_weight_set parameters, std::string path_to_output,
                                    std::string dataset_name, int num_iter);

void minimize_dense_alpha_divergence(std::string path_to_image, std::string path_to_unaries,
                                     Potts_weight_set parameters, std::string path_to_output,
                                     float alpha, std::string dataset_name);

void minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                         Potts_weight_set parameters, std::string path_to_output,
                         std::string dataset_name );

void minimize_LP(std::string path_to_image, std::string path_to_unaries,
                 Potts_weight_set parameters, std::string path_to_output,
                 std::string dataset_name);

void minimize_LR_QP(std::string path_to_image, std::string path_to_unaries,
                    Potts_weight_set parameters, std::string path_to_output,
                    std::string dataset_name);

void minimize_QP_cccp(std::string path_to_image, std::string path_to_unaries,
                      Potts_weight_set parameters, std::string path_to_output,
                      std::string dataset_name);

void gradually_minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                   Potts_weight_set parameters, std::string path_to_output,
                                   std::string dataset_name);

void minimize_cccp_mean_field(std::string path_to_image, std::string path_to_unaries,
                              Potts_weight_set parameters, std::string path_to_output,
                              std::string dataset_name);

void unaries_baseline(std::string path_to_unaries, std::string path_to_output, std::string dataset_name);

label_matrix minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                 Potts_weight_set parameters);
