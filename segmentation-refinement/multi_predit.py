import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
img_root = '/data/yliugu/front3d_ngp'
mask_root = '/data/yliugu/3dfront_2D_mask'
output_root = '/data/yliugu/3dfront_predicted_2D_mask'


scene = '3dfront_0024_01'
# image = cv2.imread('/data/yliugu/front3d_ngp/3dfront_0024_01/train/images/0001.jpg')
# mask = cv2.imread('/data/yliugu/3dfront_2D_mask/3dfront_0024_01/0001.png')

os.makedirs(output_root, exist_ok=True)

scene_mask_root = os.path.join(mask_root, scene)
scene_img_root= os.path.join(img_root, scene, 'train', 'images')
scene_output_root = os.path.join(output_root, scene + '_new')

os.makedirs(scene_output_root, exist_ok=True)

img_list = os.listdir(scene_img_root)
img_list.sort()

for i in tqdm(img_list):
    mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '.png'))
    img_file = os.path.join(scene_img_root, i)

    mask = cv2.imread(mask_file)
    image = cv2.imread(img_file)
    mask = mask.sum(axis=-1)
    instance_list = np.unique(mask)
    instance_mask_list = [] 
    for instance in instance_list:
        if instance == 0:
            continue
        instance_mask = (mask == instance).astype(np.uint8) * 255
        instance_mask_list.append(instance_mask)
    
    instance_masks = np.stack(instance_mask_list, axis=0)
    refiner = refine.Refiner(device='cuda:0') 
    output = refiner.refine(image, instance_masks[0], fast=True, L=700) 
    
    output_file = os.path.join(scene_output_root, i.replace('.jpg', f'_{instance}.png'))
    cv2.imwrite(output_file, output)
    
    # output = refiner.multi_refine(image, instance_masks, fast=True, L=700) 
    # for j in range(len(instance_list)-1):
    #     output_file = os.path.join(scene_output_root, i.replace('.', f'_{instance_list[j+1]}.'))
    #     cv2.imwrite(output_file, output[j])

