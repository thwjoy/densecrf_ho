#include <Eigen/Core>

using namespace Eigen;

struct img_size {
    int width;
    int height;
};


unsigned char* load_image(const std::string, img_size size);
MatrixXf load_unary(const std::string path_to_unary, img_size& size);
void save_map(const MatrixXf estimates, const img_size size, const std::string path_to_output);
