#include <vector>
#include <set>

#include "color_to_label.hpp"


double compute_mean_iou (const std::vector<int> & confusionMatrix, int numLabels);
double pixwise_acc_from_confmat(const std::vector<int> & confMat, int numLabels);
void evaluate_segmentation(label_matrix gt_lbls, label_matrix crf_lbls, std::vector<int>& confMat, int num_labels);
