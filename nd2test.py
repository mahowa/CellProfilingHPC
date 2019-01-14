import time
from collections import defaultdict
import numpy as np
from PIL import Image
from nd2reader import ND2Reader
from openmp_extentions import roi_avg


def roiDic2array(roi_dic):
    numPix = max((len(v)) for k, v in roi_dic.iteritems())
    num_rois = len(roi_dic)
    size = numPix * num_rois

    roi_arr_x = np.full(size, fill_value=-1, dtype=int)
    roi_arr_y = np.full(size, fill_value=-1, dtype=int)

    roi_index = 0
    for k, v in roi_dic.iteritems():
        index = 0
        for i in v:
            arrInd = roi_index*numPix + index
            index += 1
            roi_arr_x[arrInd] = i[0]
            roi_arr_y[arrInd] = i[1]
        roi_index += 1

    return roi_arr_x, roi_arr_y, numPix

start = time.time()

im = Image.open('roi.tif')
height, width = im.size
imarray = np.array(im)

rois = defaultdict(list)

for i in range(height):
    for j in range(width):
        pixel = imarray[i, j]
        if pixel != 0:
            rois[pixel].append((i, j))


rois_arr_x, rois_arr_y, roi_pixels = roiDic2array(rois)

frames = dict()

for key, value in rois.iteritems():
    frames[key] = list()

currentFrame = 1
allframes = list()

with ND2Reader('test.nd2') as images:
    # width and height of the image
    print('%d x %d px' % (images.metadata['width'], images.metadata['height']))
    print(images.metadata)

    width = images.metadata['width']
    height = images.metadata['height']

    for fov in images:
        allframes.append(np.asarray(fov))
        frame = dict()
        for key, value in rois.iteritems():
            numPixels = len(value)
            sum = 0
            for i, j in value:
                sum += fov[i, j]
            avg = float(sum)/float(numPixels)
            frames[key].append(avg)

    allframes = np.asarray(allframes, dtype=float)

    num_images = len(images)
    num_rois = len(rois)
    results = np.empty((num_rois, num_images), dtype=float)
    # Results, Images, Height, Width, # of images, rois x points, rois y point, roi # pixels per roi, # of rois
    roi_avg.get_average(results, allframes.flatten(), height, width, num_images, rois_arr_x, rois_arr_y, roi_pixels, num_rois)

vv = frames.itervalues().next()
frame_array = np.zeros((len(frames), len(vv)))

index = 0
for key, value in frames.iteritems():
    frame_array[index, :] = np.asarray(value)
    index += 1





# TODO gather on root
end = time.time()
print(end - start + " (s)")
