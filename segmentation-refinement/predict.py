import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
from tqdm import tqdm

img_root = '/data/yliugu/front3d_ngp'
mask_root = '/data/yliugu/3dfront_2D_mask_predicted'
output_root = '/data/yliugu/3dfront_refined_2D_mask_predicted'



refiner = refine.Refiner(device='cuda:0') 

# image = cv2.imread('/data/yliugu/front3d_ngp/3dfront_0024_01/train/images/0001.jpg')
# mask = cv2.imread('/data/yliugu/3dfront_2D_mask/3dfront_0024_01/0001.png')

os.makedirs(output_root, exist_ok=True)

scene_list = os.listdir(mask_root)
for scene in scene_list:
    # scene = '3dfront_0042_01'
    scene = "3dfront_0054_00"

    scene_mask_root = os.path.join(mask_root, scene)
    scene_init_mask_root = os.path.join(mask_root, scene+'_selected_1')
    scene_img_root= os.path.join(img_root, scene, 'train', 'images')
    scene_output_root = os.path.join(output_root, scene)

    os.makedirs(scene_output_root, exist_ok=True)

    img_list = os.listdir(scene_img_root)
    img_list.sort()

    for i in tqdm(img_list):
    
        mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '_l.png'))
        init_mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '_s.png'))
        img_file = os.path.join(scene_img_root, i)

        mask = cv2.imread(mask_file)[..., 0]
        sparse_mask = cv2.imread(init_mask_file)[..., 0]
        image = cv2.imread(img_file)
        instance_list = np.unique(mask)
        for instance in instance_list:
            if instance == 0:
                continue
            instance_mask = (mask == instance).astype(np.uint8) * 255
            inital_mask = (sparse_mask == instance).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            
            # weight = 0.5
            # end_weight = 0.4
            # iter = 15
            # weight_decrease = (weight - end_weight) / iter
            # for j in range(iter):
            #     output = refiner.refine(image, instance_mask, fast=False, L=700) 
            #     instance_mask = inital_mask * weight + output * (1-weight) + 1
            #     weight = weight - weight_decrease
            #     instance_mask = cv2.dilate(instance_mask, kernel, iterations=1)
            output = refiner.refine(image, instance_mask, fast=False, L=700) 
            output_file = os.path.join(scene_output_root, i.replace('.jpg', f'_{instance}.png'))
            cv2.imwrite(output_file, output)

