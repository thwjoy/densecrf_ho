#include <iostream>
#include <string>
#include "msImageProcessor.h"
#include "libppm.h"


int main(int argc, char ** argv) {
	std::string file_name;
	int width, height, depth;
	depth = 3;


	if (argc < 2) {
		std::cout << "Please provide a file!" << std::endl;
		return 0;
	} 

	unsigned char * image_data = readBMPImage(argv[1], height, width);
	unsigned char * output_data = new unsigned char[width*height*depth];
/*
	msImageProcessor m_process;
	m_process.DefineImage(image_data,COLOR,height,width);
	m_process.Segment(8,4,500,NO_SPEEDUP);
	m_process.GetResults(output_data);
	printf("1:%d\r\n",image_data[0]);
	printf("2:%d\r\n",image_data[1]);
	printf("3:%d\r\n",image_data[2]);*/
	writePPMImage("./images/output.ppm",image_data,height,width,depth,"");



}