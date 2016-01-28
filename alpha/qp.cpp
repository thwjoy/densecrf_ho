#include "qp.hpp"


void descent_direction(Eigen::MatrixXf & out, const Eigen::MatrixXf & grad){
    out.resize(grad.rows(), grad.cols());
    out.fill(0);
    int M =  grad.rows();
    int N =  grad.cols();

    int m;
    for (int i=0; i < N; i++) {
        grad.col(i).maxCoeff(&m);
        out(m,i) = 1;
    }
}
