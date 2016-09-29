
#pragma once
#include "objective.h"

using namespace Eigen;

// run max-flow on W x H image, binary densecrf problem with gaussian pairwise weights 
// returns maxflow and resulting labels in labels
double testIBFS(VectorXs & labels, int H, int W, const MatrixXf & unaries, float sigma, float weight);

