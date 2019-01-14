extern int numImages;
extern int width;
extern int numRois;
extern int roiPixels;
void get_average(double * frames, int frame_size, double * images,  int imgs_size,  double * rois_x, int rois_x_size, double * rois_y,  int rois_y_size);
void setVars(int _numImages, int _width, int _numRois, int _roiPixels);
void setNumThreads(int num_threads);


