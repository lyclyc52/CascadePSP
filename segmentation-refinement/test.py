import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np

image = cv2.imread('/data/yliugu/front3d_ngp/3dfront_0024_01/train/images/0001.jpg')
mask = cv2.imread('/data/yliugu/3dfront_2D_mask/3dfront_0024_01/0001.png')


# width = int(image.shape[1] * 4)
# height = int(image.shape[0] * 4)
# dim = (width, height)
# image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

mask = mask.sum(axis=-1)
temp = np.unique(mask)
mask = (mask == temp[1]).astype(np.uint8) * 255
# mask = cv2.imread('test/aeroplane.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('test/aeroplane.jpg')

# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=False, L=900) 

cv2.imwrite('./test/output_1.jpg', output)
# plt.imshow(output)
# plt.show()
