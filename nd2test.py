from nd2reader import ND2Reader
import matplotlib.pyplot as plt
from PIL import Image
import numpy
from collections import defaultdict
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start = time.time()
if rank == 0:
    im = Image.open('roi.tif')
    height, width = im.size
    imarray = numpy.array(im)

    rois = defaultdict(list)

    for i in range(height):
        for j in range(width):
            pixel = imarray[i, j]
            if pixel != 0:
                rois[pixel].append((i, j))
else:
    rois = None

rois = comm.bcast(rois, root=0)

# TODO test speed of these two
# for i in range(height):
#     for j in range(width):
#         pixel = im.getpixel((i, j))
#         if pixel != 0:
#             rois[pixel].append((i, j))



frames = list()
currentFrame = 1
if rank == 0:
    with ND2Reader('test.nd2') as images:
        # width and height of the image
        print('%d x %d px' % (images.metadata['width'], images.metadata['height']))
        print(images.metadata)
else:
    images = 0

images = comm.scatter(images, root=0)

for fov in images:
    frame = dict()
    for key, value in rois.iteritems():
        numPixels = len(value)
        sum = 0
        for i, j in value:
            sum += fov[i, j]
        avg = sum/numPixels
        frame[key] = avg
    frames.append(frame)

# TODO gather on root
end = time.time()
print(end - start)


