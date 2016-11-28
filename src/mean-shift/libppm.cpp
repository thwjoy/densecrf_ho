////////////////////////////////////////////////////////
// Name     : libppm.cpp
// Purpose  : Read/Write Portable Pixel Map images
// Author   : Chris M. Christoudias
// Modified by
// Created  : 03/20/2002
// Copyright: (c) Chris M. Christoudias
// Version  : v0.1
////////////////////////////////////////////////////////

#include "libppm.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>

int writePPMImage(char *filename, unsigned char *image, int height, int width, int depth, char *comments)
{

  if(!filename || !image) return PPM_NULL_PTR;
  FILE *fp = fopen(filename, "wb");
  if(!fp) return PPM_FILE_ERROR;

  //********************************************************
  //Write header information and comments.
  //********************************************************

  fprintf(fp, "P6\n", width, height);
  if(comments && strlen(comments) <= 70) fprintf(fp, "%s\n", comments);
  fprintf(fp, "%d %d\n%d\n", width, height, 255);
  
  //********************************************************
  //Output raw image data.
  //********************************************************

  //for some reason the fwrite would shift the image data along one, hence to avoid this, ignore the first value
  int writeCount = fwrite(image, 3 * sizeof(unsigned char), height*width*3, fp);
  fclose(fp);
  if(writeCount !=height*width*3) return PPM_FILE_ERROR;
  return PPM_NO_ERRORS;
}


unsigned char * readPPMImage(char *filename, int& height, int& width, int& depth)
{

  FILE * file = fopen(filename,"rb");
  if (file == NULL) {
    //throw an exception
    throw std::runtime_error("Error unable to open file");
  }

  char p, n;
  int max_val;
  //we now need to get the dimensions of the file, fortunately this is very easily done with a ppm file
  fscanf(file, "%c %c %d %d %d", &p, &n, &width, &height, &max_val);
  if (p != 'P') { //this is not a ppm file!
    //throw an exception
    throw std::runtime_error("This is not a PPM file! AHHHHHHHHH");
  }

  //could probably get away with using a raw pointer but why run the risk
  unsigned char * image = new unsigned char[width*height*depth];
  
  short pixel_size = 1;
  if (max_val > 255) {
    throw std::runtime_error("Error, the code is only able to deal with pixels of 8 bits");
  }

  if (n == '6') { //is not plain type ppm file
    fread(image,3 * sizeof(unsigned char),width*height*depth,file);
  } else if (n == '3') {
    int pixel;
    for ( int i=0; i<(width*height*depth); i++ )
    {
      fscanf ( file, "%d", &pixel );
      image[i] = pixel;     
    }
  }

  return image;

}

unsigned char* readBMPImage(char* filename, int& height, int& width)
{
    int i;
    FILE* f = fopen(filename, "rb");

    if(f == NULL)
        throw "Argument Exception";

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    width = *(int*)&info[18];
    height = *(int*)&info[22];

    std::cout << *(short*)&info[30] << std::endl;


    int row_padded = (width*3 + 3) & (~3);
    unsigned char* data = new unsigned char[row_padded];
    unsigned char tmp;

    //will read in the alpha channel!!!
    for(int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, f);

    }

    fclose(f);
    return data;
}


