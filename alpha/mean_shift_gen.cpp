#include "msImageProcessor.h"
#include "file_storage.hpp"
#include <vector>
#include <fstream>
#include <iostream>


int main (int argc, char ** argv) {

    std::string path_to_data = "/media/tom/DATA/datasets/MSRC";
    std::vector<std::string> imgs;

    //get image names
    std::string path_to_split = path_to_data + "/split/Test_acc.txt";
    std::ifstream file(path_to_split.c_str());
    std::string next_img_name;
    while(getline(file, next_img_name)){
        imgs.push_back(next_img_name);
        std::cout << next_img_name << std::endl;
    }

    file.close();
    int regions[] = {100,250,400};
    for (const auto & image_name : imgs)
    {
        for (const auto & reg : regions)
        {
            img_size size = {-1, -1};
            unsigned char * img = load_image(path_to_data + "/Images/" + image_name + ".bmp", size);
            unsigned char * segment_image = new unsigned char[size.width * size.height * 3];
            std::vector<int> regions_out;
            std::vector<std::vector<double>> super_pixel_container;

            int region;

            std::cout << "Generating segmentations for " << image_name << std::endl;

            //get the mean shift info
            msImageProcessor m_process;
            m_process.DefineImage(img , COLOR , size.height , size.width);
            m_process.Segment(8,4,reg,NO_SPEEDUP);
            m_process.GetResults(segment_image);
            int num_super_pixels = m_process.GetRegions(regions_out);

            //now write to a binary file

            std::string classifier_name = path_to_data + "/SuperPixels/" + std::to_string(reg) + "/" + image_name + "_clsfr.bin";
            std::fstream classifierOutput(classifier_name, std::ios_base::binary | std::ios_base::out);
            classifierOutput.write(reinterpret_cast<const char *>(&regions_out[0]), regions_out.size() * sizeof(int));
            classifierOutput.close();


            std::vector<int> temp;
            temp.resize(size.width * size.height);

            std::fstream bin_in(classifier_name,std::ios_base::binary|std::ios_base::in);
            bin_in.seekp(0);
            bin_in.seekg(0);
            bin_in.read((char *) temp.data(), temp.size() * sizeof(int));
            bin_in.close();

            for (int i = 0; i < temp.size(); i++)
            {
                if (i % 1000 == 0) std::cout << temp[i] << std::endl;
            }


            delete[] img;
            delete[] segment_image;

        }
    }


    return 1;
}
