//        for fov in local_images:
//            for key, value in rois.iteritems():
//                frames[key].append(compute_avg(value, fov))
//def compute_avg(val, frm):
//    numPixels = len(val)
//    sum = 0
//    for i, j in val:
//        sum += frm[i, j]
//    avg = double(sum) / double(numPixels)
//    return avg

#include <omp.h>
#include <stdio.h>
#include "roi_avg.h"

int numImages = 0;
int width = 0;
int numRois = 0;
int roiPixels = 0;

//Set global vars
void setVars(int _numImages, int _width, int _numRois, int _roiPixels){
    numImages = _numImages;
    width = _width;
    numRois =_numRois;
    roiPixels = _roiPixels;

    //FOR DEBUGGING
//    int num_threads = omp_get_thread_num();
//    printf("Num Images %d, width %d, roiPixels %d, numRois %d, num_threads %d\n", numImages, width, roiPixels, numRois, num_threads);
}

// Set number of threads for openmp
void setNumThreads(int num_threads){
//    omp_set_num_threads(num_threads);
    num_threads = omp_get_thread_num();
    printf("Using %d threads per process\n", num_threads);
}

//TODO check for bug (might not be calculating 100% correctly)
void get_average(double * frames, int frame_size, double * images,  int imgs_size,  double * rois_x, int rois_x_size, double * rois_y,  int rois_y_size){
    int imageSize = width * width;

#pragma omp parallel for
    for(int i = 0; i < numImages; i++){
        int startIndex = i*imageSize;
    #pragma omp parallel for
        for(int j = 0; j < numRois; j++){
           int roiSIndex = j* roiPixels;
           int sum = 0;
           int count = 0;
        #pragma omp parallel for reduction(+:sum)
           for(int k = 0; k < roiPixels; k++){
                int roiIndex = roiSIndex + k;

                if(rois_x[roiIndex] == -1) continue;

                int currentIndex = startIndex + (rois_y[roiIndex]*width + rois_x[roiIndex]);

                sum += images[currentIndex];
                count++;
           }

           double avg = (double)sum/(double)count;
           frames[i* numImages + j] = avg;
        }
    }


}






