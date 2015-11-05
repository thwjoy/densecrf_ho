#include <Eigen/Core>

using namespace Eigen;

// Height, Width
typedef std::pair<int,int> img_size;


MatrixXf loadUnary(const std::string path_to_unary, img_size& size);
void save_map(const MatrixXf estimates, const img_size size, const std::string path_to_output);
