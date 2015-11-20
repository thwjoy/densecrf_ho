#include <opencv2/opencv.hpp>
#include <unordered_map>

struct vec3bcomp{
    bool operator() (const cv::Vec3b& lhs, const cv::Vec3b& rhs) const
        {
            for (int i = 0; i < 3; i++) {
                if(lhs[i]!=rhs[i]){
                    return lhs.val[i]<rhs.val[i];
                }
            }
            return false;
        }
};

typedef std::map<cv::Vec3b, int, vec3bcomp> labelindex;
labelindex  init_map();
int lookup_label_index(labelindex color_to_label, cv::Vec3b gtVal);
