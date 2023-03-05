import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
from tqdm import tqdm

img_root = '/data/yliugu/front3d_ngp'
mask_root = '/data/yliugu/3dfront_2D_mask_predicted'
output_root = '/data/yliugu/output_check'



# 125, 128

# image = cv2.imread('/data/yliugu/front3d_ngp/3dfront_0024_01/train/images/0001.jpg')
# mask = cv2.imread('/data/yliugu/3dfront_2D_mask/3dfront_0024_01/0001.png')

mask_file = '/data/yliugu/3dfront_2D_mask_predicted/3dfront_0054_00_selected/0128_8.png'
mask_file_1 = '/data/yliugu/3dfront_2D_mask_predicted/3dfront_0054_00_selected_1/0128_8.png'
# mask_file = '/data/yliugu/output_check/output.png'
img_file = '/data/yliugu/front3d_ngp/3dfront_0054_00/train/images/0128.jpg'

mask = cv2.imread(mask_file)

image = cv2.imread(img_file)

refiner = refine.Refiner(device='cuda:1') 

mask = mask[..., 0]
inital_mask = cv2.imread(mask_file_1)[..., 0]
kernel = np.ones((5, 5), np.uint8)
weight = 0.5
end_weight = 0.4
iter = 10 
weight_decrease = (weight - end_weight) / iter
for j in range(iter):
    output = refiner.refine(image, mask, fast=False, L=700) 
    mask = inital_mask * weight + output * (1-weight) + 1
    weight = weight - weight_decrease
    mask = cv2.dilate(mask, kernel, iterations=1)
    output_file = os.path.join(output_root, f'output_2_{j}.png')
    cv2.imwrite(output_file, output)

