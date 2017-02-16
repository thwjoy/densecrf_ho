////////////////////////////////////////////////////////
// Name     : libppm.h
// Purpose  : Read/Write Portable Pixel Map images
// Author   : Chris M. Christoudias
// Modified by
// Created  : 03/20/2002
// Copyright: (c) Chris M. Christoudias
// Version  : v0.1
////////////////////////////////////////////////////////

#ifndef LIBPPM_H
#define LIBPPM_H

//define error constants
enum {
  PPM_NO_ERRORS,
  PPM_NULL_PTR,
  PPM_FILE_ERROR,
  PPM_UNKNOWN_FORMAT,
  PPM_OUT_OF_MEMORY
};

int writePPMImage(char *filename, unsigned char *image, int height, int width, int depth, char *comments);
unsigned char * readPPMImage(char *filename, int& height, int& width, int& depth);
unsigned char * readBMPImage(char *filename, int& height, int& width);


#endif
