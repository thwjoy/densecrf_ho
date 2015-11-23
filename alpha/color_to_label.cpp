#include "color_to_label.hpp"


labelindex  init_color_to_label_map(){
    labelindex color_to_label;

    color_to_label[cv::Vec3b(128,0,0)] = 0;
    color_to_label[cv::Vec3b(0,128,0)] = 1;
    color_to_label[cv::Vec3b(128,128,0)] = 2;
    color_to_label[cv::Vec3b(0,0,128)] = 3;
    color_to_label[cv::Vec3b(0,128,128)] = 4;
    color_to_label[cv::Vec3b(128,128,128)] = 5;
    color_to_label[cv::Vec3b(192,0,0)] = 6;
    color_to_label[cv::Vec3b(64,128,0)] = 7;
    color_to_label[cv::Vec3b(192,128,0)] = 8;
    color_to_label[cv::Vec3b(64,0,128)] = 9;
    color_to_label[cv::Vec3b(192,0,128)] = 10;
    color_to_label[cv::Vec3b(64,128,128)] = 11;
    color_to_label[cv::Vec3b(192,128,128)] = 12;
    color_to_label[cv::Vec3b(0,64,0)] = 13;
    color_to_label[cv::Vec3b(128,64,0)] = 14;
    color_to_label[cv::Vec3b(0,192,0)] = 15;
    color_to_label[cv::Vec3b(128,64,128)] = 16;
    color_to_label[cv::Vec3b(0,192,128)] = 17;
    color_to_label[cv::Vec3b(128,192,128)] = 18;
    color_to_label[cv::Vec3b(64,64,0)] = 19;
    color_to_label[cv::Vec3b(192,64,0)] = 20;
    color_to_label[cv::Vec3b(0,0,0)] = 21;

    // Ignored labels
    color_to_label[cv::Vec3b(64,0,0)] = 21;
    color_to_label[cv::Vec3b(128,0,128)] = 21;

    return color_to_label;
}

int lookup_label_index(labelindex color_to_label, cv::Vec3b gtVal)
{
    int label=-1;
    try {
        label = color_to_label.at(gtVal);
    } catch( std::out_of_range) {
        //std::cout << gtVal << '\n';
        (void)0;
    }
    if (label != -1) {
        return label;
    } else {
        return 21;
    }
}
