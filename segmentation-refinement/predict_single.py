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

mask_file = '/data/yliugu/3dfront_2D_mask_predicted/3dfront_0054_00_selected_1/0128_8.png'
# mask_file = '/data/yliugu/output_check/output.png'
img_file = '/data/yliugu/front3d_ngp/3dfront_0054_00/train/images/0128.jpg'

mask = cv2.imread(mask_file)
image = cv2.imread(img_file)

refiner = refine.Refiner(device='cuda:0') 

mask = mask[..., 0]
inital_mask = mask
kernel = np.ones((7, 7), np.uint8)
for j in range(20):
    output = refiner.refine(image, mask, fast=False, L=700) 

    output = cv2.dilate(output, kernel, iterations=2)
    mask = (inital_mask + output) / 2 +1
    output_file = os.path.join(output_root, f'output_1_{j}.png')
    cv2.imwrite(output_file, output)

