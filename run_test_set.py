#!/usr/bin/env python
import os
#get the list off all images in the 

unaries = os.listdir("/home/tomj/Documents/4YP/densecrf/data/MSRC/texton_unaries")


for file in unaries:
	sub_str = file.split('.',1);
	execute = "build/alpha/non_convex " + sub_str[0]
	os.system(execute);
