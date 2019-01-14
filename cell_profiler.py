#############################################################################
# Cell Profiler
#
# Author: Matt Howa
# Description: Final project for University of Utah HPC class in FA 2018
# Takes a input file defining ROI (cells) positions along with a video file in
# nd2 file format. It then calculates each ROI mean intensity for each frame of the video.
# The program then outputs the results to a csv file where the rows are individual ROIs
# and the columns are the ROI's mean intensity by each frame in the respective order provided
#############################################################################

# See readme on how to properly install each of the inputs below correctly
import sys
import time
from collections import defaultdict
import math
import numpy as np
from PIL import Image
from mpi4py import MPI
from nd2reader import ND2Reader
from openmp_extentions import roi_avg


# Before adding OpenMP support through swig and C++
# This function accurately calculates the mean intensity of each ROI
def compute_avg(val, frm):
    numPixels = len(val)
    sum = 0
    for i, j in val:
        sum += frm[i, j]
    avg = float(sum) / float(numPixels)
    return avg


# Converts a dictionary to a nXn array
def dict2array(dictionary):
    dict_array = np.zeros((len(dictionary), len(dictionary.itervalues().next())))
    index = 0
    for key, value in dictionary.iteritems():
        dict_array[index, :] = np.asarray(value)
        index += 1
    return dict_array


# Takes the roi dictionary and splits it into x an y components both of size nXn
def roiDic2arrays(roi_dic):
    numPix = max((len(v)) for k, v in roi_dic.iteritems())
    num_rois = len(roi_dic)
    size = numPix * num_rois

    roi_arr_x = np.full(size, fill_value=-1, dtype=float)
    roi_arr_y = np.full(size, fill_value=-1, dtype=float)

    roi_index = 0
    for k, v in roi_dic.iteritems():
        index = 0
        for i in v:
            arrInd = roi_index * numPix + index
            index += 1
            roi_arr_x[arrInd] = i[0]
            roi_arr_y[arrInd] = i[1]
        roi_index += 1

    return roi_arr_x, roi_arr_y, numPix


# This is the main function
# Reads in images from a file  #TODO allow this as input
# Reads in roi image from file #TODO all this as input
# Calculates mean intensity of each ROI in each image
# Outputs file to results as csv #todo possibly allow input of output file name
# TODO needs to output file that has ROI initial lables ?? maybe
# TODO there is either a bug in this code or in the roi_avg.cpp code that is causing the values to be calculated wrong
# TODO Must address for use in production, however for timing results and final project submission it should be adequate
def run(omp_threads):
    roi_avg.setNumThreads(int(omp_threads))

    comm = MPI.COMM_WORLD
    # print("%d of %d" % (comm.Get_rank(), comm.Get_size()))  # For Debugging

    size = comm.Get_size()
    rank = comm.Get_rank()

    start = time.time()
    if rank == 0:
        im = Image.open('roi.tif')  # TODO might want to make this input to the program
        height, width = im.size
        imarray = np.array(im)

        rois = defaultdict(list)

        for i in range(height):
            for j in range(width):
                pixel = imarray[i, j]
                if pixel != 0:
                    rois[pixel].append((i, j))
    else:
        rois = None

    # Broadcast ROI locations
    rois = comm.bcast(rois, root=0)

    # Convert roi dictionary to nXn matrix for x and y  values and output n
    rois_arr_x, rois_arr_y, roi_pixels = roiDic2arrays(rois)

    frames = dict()
    for key, value in rois.iteritems():
        frames[key] = list()

    # Using library nd2 reader, read in file to start computation
    with ND2Reader('video.nd2') as images:  # TODO might want to make this input to the program
        # width and height of the image are the same #todo might want to add support for non square images
        width = images.metadata['width']

        imgs_per_proc = int(math.floor(float(len(images)) / float(size)))  # Work to be done per MPI process
        total_imgs = int(imgs_per_proc * size)  # Total work that can be done in parallel (MPI)

        startIndex = int(rank * imgs_per_proc)  # Start Index to get local work
        endIndex = int(startIndex + imgs_per_proc)  # End index of local work

        local_images = images[startIndex:endIndex]  # Work to be done on process (MPI)

        # print("%d to %d on %d" % (startIndex, endIndex, rank)) #For Debugging

        # Old code needed before using OpenMP with swig
        # for fov in local_images:
        #     for key, value in rois.iteritems():
        #         frames[key].append(compute_avg(value, fov))
        # frame_array = dict2array(frames)

        allframes = list()  # a list for numpy arrays
        for fov in local_images:  # get each image as matrix
            allframes.append(np.asarray(fov))
        allframes = np.asarray(allframes, dtype=float)  # convert list to numpy
        num_rois = len(rois)  # number of ROIs found from input file
        allframes = allframes.flatten()  # flatten matrix to 1D to send to c++ extension file

        # C++ Extension in order to use OpenMP
        roi_avg.setVars(imgs_per_proc, width, num_rois, roi_pixels)
        results = roi_avg.get_average(int((num_rois * imgs_per_proc)), allframes, rois_arr_x, rois_arr_y)

        frame_array = results

        recv_buf = None  # Receive buffer to gather all calculations on root
        if rank == 0:
            flat_size = int((num_rois * imgs_per_proc) * size)  # The total amount to get from processes
            recv_buf = np.zeros(flat_size, dtype=float)  # Initialize buffer size

        comm.Gather(frame_array, recv_buf, root=0)  # Gather on root

        # Do rest of work in serial on root
        if rank == 0:
            frame_arrays = list()  # List for each processes work
            flat_size = num_rois * imgs_per_proc  # Size of data per process
            # Break results into per process size and add to list
            for i in range(size):
                startIndex = int(i * flat_size)
                endIndex = int(startIndex + flat_size)
                frame_arrays.append(np.reshape(recv_buf[startIndex:endIndex], (num_rois, imgs_per_proc)))

            # Now get the data into the right form for output
            index = 0
            for key, value in rois.iteritems():
                for i in range(size):
                    frames[key].extend(frame_arrays[i][index, :])
                index += 1
            #########
            # Do it again on the remaining images that could not be broken up evenly among other processes
            # TODO might be able to hide a bit of time by doing this in parallel with some thoughtful work but
            # TODO im not sure itll speed up by much
            remaining_images = images[total_imgs:]
            imgs_left = len(remaining_images)
            roi_avg.setVars(imgs_left, width, num_rois, roi_pixels)
            results = roi_avg.get_average(int((num_rois * imgs_left)), allframes, rois_arr_x, rois_arr_y)
            frame = np.reshape(results, (num_rois, imgs_left))
            index = 0
            for key, value in rois.iteritems():
                frames[key].extend(frame[index, :])
                index += 1
            #########
            # Old code needed before using OpenMP with swig
            # for fov in remaining_images:
            #     for key, value in rois.iteritems():
            #         frames[key].append(compute_avg(value, fov))

            results = dict2array(frames)
            import datetime
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            outfile = "Results/Run_at_" + st + ".csv"  # TODO might want to make this an input to the program
            np.savetxt(outfile, results, delimiter=',')

    if rank == 0:
        end = time.time()
        print(str(end - start) + " (s)")
        print("Results will be found in " + outfile)


#############################################################################
# run the program
#############################################################################
if len(sys.argv) != 2:
    print("You didn't provide the correct number of arguments ")
    exit()

omp_threads_input = sys.argv[1]
run(omp_threads_input)
