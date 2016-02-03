#include "file_storage.hpp"


void minimize_dense_alpha_divergence(std::string path_to_image, std::string path_to_unaries,
                                     std::string path_to_parameters,  std::string path_to_output,
                                     float alpha, std::string dataset_name);

void minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                         std::string path_to_output, std::string path_to_parameters,
                         std::string dataset_name );

void minimize_LR_QP(std::string path_to_image, std::string path_to_unaries,
                    std::string path_to_output, std::string path_to_parameters,
                    std::string dataset_name);

void minimize_QP_cccp(std::string path_to_image, std::string path_to_unaries,
                      std::string path_to_output, std::string path_to_parameters,
                      std::string dataset_name);

void gradually_minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                   std::string path_to_output, std::string path_to_parameters,
                                   std::string dataset_name);

void minimize_cccp_mean_field(std::string path_to_image, std::string path_to_unaries,
                              std::string path_to_output, std::string path_to_parameters,
                              std::string dataset_name);

void unaries_baseline(std::string path_to_unaries, std::string path_to_output, std::string dataset_name);

label_matrix minimize_mean_field(std::string path_to_image, std::string path_to_unaries,
                                 float w1, float sigma_alpha, float sigma_beta);
