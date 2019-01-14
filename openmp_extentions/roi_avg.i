
%module roi_avg

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_CAST_MODE
#include "roi_avg.h"

%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* frames, int frame_size)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* images, int imgs_size), (double* rois_x, int rois_x_size), (double* rois_y, int rois_y_size)}
void get_average(double * frames, int frame_size,  double * images,  int imgs_size,  double * rois_x, int rois_x_size, double * rois_y,  int rois_y_size);
int numImages;
int width;
int numRois;
int roiPixels;
void setVars(int _numImages, int _width, int _numRois, int _roiPixels);
void setNumThreads(int num_threads);