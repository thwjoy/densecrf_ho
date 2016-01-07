#include <limits>
#include "evaluation.hpp"

///////////////////////////////////
// Confusion Matrix manipulation //
///////////////////////////////////

void sum_along_row (const std::vector<int> & matrix, int n_rows, int n_cols, std::vector<int>& sums){
    sums.clear();
    sums.assign(n_rows, 0);

    for (size_t row = 0; row < n_rows; ++row){
        for (size_t col = 0; col < n_cols; ++col){
            sums[row] += matrix[row*n_rows + col];
        }
    }
}

void sum_along_col (const std::vector<int> & matrix, int n_rows, int n_cols, std::vector<int>& sums){
    sums.clear();
    sums.assign(n_cols, 0);

    for (size_t col = 0; col < n_cols; ++col){
        for (size_t row = 0; row < n_rows; ++row){
            sums[col] += matrix[row*n_rows + col];
        }
    }
}

template <typename T>
double mean_vector (const std::vector<T> & vector, std::set<int> & indicesNotConsider)
{
    double mean = 0;
    double N = 0;

    for (size_t i = 0; i < vector.size(); ++i){
        if ( indicesNotConsider.find(i) == indicesNotConsider.end() ){
            mean = mean + ( (double) vector[i] );
            ++N;
        }
    }

    return mean/N;
}


void find_blank_gt (const std::vector<int> & rowSums, std::set<int> & indicesNotConsider)
{
    for (int i = 0; i < rowSums.size(); ++i){
        if (rowSums[i] == 0) {
            indicesNotConsider.insert(i);
        }
    }
}


double compute_mean_iou (const std::vector<int> & confusionMatrix, int numLabels){
    std::set<int> indicesNotConsider;
    indicesNotConsider.insert(numLabels-1); // no void labels

    std::vector<int> rowSums;
    std::vector<int> colSums;

    std::vector<double> iouPerClass;
    iouPerClass.clear();

    sum_along_row (confusionMatrix, numLabels, numLabels, rowSums);
    sum_along_col (confusionMatrix, numLabels, numLabels, colSums);

    for (size_t i = 0; i < numLabels; ++i){
        size_t uni = rowSums[i] + colSums[i] - confusionMatrix[numLabels * i + i]; // "union" is a c++ reserved word
        size_t intersection = confusionMatrix[numLabels * i + i];
        double iou = intersection/ ((double) (uni + std::numeric_limits<double>::epsilon() ));

        iouPerClass.push_back(iou);
    }
    // Necessary because in the case of a label not present in a GT,
    // this would mean an IoU of zero which is exagerated
    find_blank_gt (rowSums, indicesNotConsider);
    double mean = mean_vector(iouPerClass, indicesNotConsider);

    return mean;
}


double pixwise_acc_from_confmat(const std::vector<int> & confMat, int numLabels){
    double nb_of_total_pixels = 0;
    double nb_of_correct_pixels = 0;

    for (int row=0; row < numLabels; row++) {
        for (int col=0; col < numLabels; col++) {
            if (col==numLabels-1) { // Ignore the "no classification" error
                continue;
            }
            if (row==numLabels-1) {
                continue;
            }
            nb_of_total_pixels += confMat[row * numLabels+ col];
            if(row==col){
                nb_of_correct_pixels += confMat[row * numLabels + col];
            }
        }
    }
    return 100* nb_of_correct_pixels / nb_of_total_pixels;
}

void evaluate_segmentation(label_matrix gt_lbls, label_matrix crf_lbls, std::vector<int>& confMat, int num_labels) {
    assert(gt_lbls.size() == crf_lbls.size());
    for(int y = 0; y < gt_lbls.size(); ++y)
    {
        assert(gt_lbls[y].size() == crf_lbls[y].size());
        for(int x = 0; x < gt_lbls[y].size(); ++x)
        {
            int gt_index = gt_lbls[y][x];
            int crf_index = crf_lbls[y][x];
            ++confMat[gt_index * num_labels + crf_index];
        }
    }

}
