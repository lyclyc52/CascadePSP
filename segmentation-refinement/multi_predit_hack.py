import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
import torch
from tqdm import tqdm

Use_hack = False
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

img_root = '/data/yliugu/front3d_ngp'
mask_root = '/data/yliugu/3dfront_2D_mask_predicted'
output_root = '/data/yliugu/3dfront_refined_2D_mask_predicted'

refiner = refine.Refiner(device='cuda:0') 

os.makedirs(output_root, exist_ok=True)

batch_size = 5


im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])

scene_list = os.listdir(mask_root)

kernel = np.ones((5, 5), np.uint8)
weight = 0.5
end_weight = 0.4
iter = 10 
weight_decrease = (weight - end_weight) / iter
for scene in scene_list:
    scene = "3dfront_0054_00"

    scene_mask_root = os.path.join(mask_root, scene)
    scene_img_root= os.path.join(img_root, scene, 'train', 'images')
    scene_output_root = os.path.join(output_root, scene)
    os.makedirs(scene_output_root, exist_ok=True)

    img_list = os.listdir(scene_img_root)
    img_list.sort()
    for i in tqdm(img_list):
        # mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '.png'))
        img_file = os.path.join(scene_img_root, i)
        mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '_l.png'))
        init_mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '_s.png'))

        mask = cv2.imread(mask_file)[..., 0]
        image = cv2.imread(img_file)
        sparse_mask = cv2.imread(init_mask_file)[..., 0]

        instance_list = np.unique(mask)
        instance_mask_list = [] 
        init_mask_list = []
        for instance in instance_list:
            if instance == 0:
                continue
            instance_mask_list.append((mask == instance).astype(np.uint8) * 255)
            init_mask_list.append((sparse_mask == instance).astype(np.uint8) * 255)
        
        instance_masks = np.stack(instance_mask_list, axis=0)
        init_mask = np.stack(init_mask_list, axis=0)
        
        
        if Use_hack:
            for j in range(iter):
                output = refiner.multi_refine(image, instance_masks, fast=False, L=700) 
                mask = init_mask * weight + output * (1-weight) + 1
                weight = weight - weight_decrease
                instance_masks = []
                for k in range(mask.shape[0]):
                    instance_masks.append(cv2.dilate(mask[k], kernel, iterations=1))
                instance_masks = np.stack(instance_masks , 0)
                # output_file = os.path.join(output_root, f'output_2_{j}.png')
                # cv2.imwrite(output_file, output)
        else:
            output = refiner.refine(image, instance_masks[0], fast=False, L=700) 
            
        for j in range(len(instance_list)):
            if instance == 0:
                continue
            output_file = os.path.join(scene_output_root, i.replace('.jpg', f'_{j}.png'))
        
        cv2.imwrite(output_file, output[j - 1])

    exit()  
    
    # all_images = []
    # all_instance_mask = []
    # all_init_mask = []
    # all_instance_list = []
    # all_image_name = []
    # print('load images:')
    # for i in tqdm(img_list):
    #     mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '_l.png'))
    #     init_mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '_s.png'))
    #     img_file = os.path.join(scene_img_root, i)

    #     mask = cv2.imread(mask_file)[..., 0]
    #     sparse_mask = cv2.imread(init_mask_file)[..., 0]
    #     image = cv2.imread(img_file)
    #     instance_list = np.unique(mask)
        
    #     instance_mask = []
    #     init_mask = []
    #     for instance in instance_list:
    #         if instance == 0:
    #             continue
    #         instance_mask.append((mask == instance).astype(np.uint8) * 255)
    #         init_mask.append((sparse_mask == instance).astype(np.uint8) * 255)
    #     kernel = np.ones((5, 5), np.uint8)
    #     instance_mask = np.stack(instance_mask, axis=0)
    #     init_mask = np.stack(init_mask, axis=0)
        
    #     all_images.append(image)
    #     all_instance_mask.append(instance_mask)
    #     all_init_mask.append(init_mask)
    #     all_instance_list.append(instance_list)
    #     all_image_name.append(i)
    
    
    
    # batch_split = range(len(all_images))
    # batch_split = np.array_split(batch_split, batch_size)
    
    
            
    # weight = 0.5
    # end_weight = 0.4
    # iter = 15
    # weight_decrease = (weight - end_weight) / iter
        
        
    # for b in range(0, len(all_images), batch_size):
    #     image = all_images[b: b+batch_size]
    #     instance_mask = all_instance_mask[b: b+batch_size]
    #     init_mask = all_init_mask[b: b+batch_size]
    #     instance_list = all_instance_list[b: b+batch_size]
    #     image_name = all_image_name[b: b+batch_size]
        
    #     image = np.stack(image, 0)
        
    #     mask_split = [0]
    #     for j in range(len(instance_mask)):
    #         mask_split.append(instance_mask[j].shape[0] + mask_split[j] )
            
    #     instance_mask = np.concatenate(instance_mask, 0)
    #     init_mask = np.concatenate(init_mask, 0)
    #     print(instance_mask.shape)
        
    #     im_list = []
    #     for t in range(image.shape[0]):
    #         temp = im_transform(image[t]).unsqueeze(0)
    #         temp  = temp.expand([mask_split[t+1] - mask_split[t], -1,-1,-1])
    #         im_list.append(temp)
            
    #     image = torch.cat(im_list, 0)
    #     print(image.shape)
    #     temp_list = []

    #     for j in range(iter):
    #         mask_list = []
    #         for t in range(image.shape[0]):
    #             mask_list.append(seg_transform((instance_mask[t]>127).astype(np.uint8)*255))
    #         instance_mask = torch.cat(mask_list)
            
            
    #         output = refiner.multi_refine(image, instance_mask, fast=False, L=700) 
    #         instance_mask = init_mask * weight + output * (1-weight) + 1
    #         weight = weight - weight_decrease
    #         instance_mask = cv2.dilate(instance_mask, kernel, iterations=1)
            
    #     for j in range(len(image_name)):
    #         if j == 0:
    #             continue
    #         im_output = output[mask_split[j]: mask_split[j+1]]
    #         im_instance_list = instance_list[mask_split[j]: mask_split[j+1]]
    #         im_name = image_name[j]
    #         for k in range(len(im_instance_list)):
    #             output_file = os.path.join(scene_output_root, im_name.replace('.jpg', f'_{im_instance_list[k]}.png'))
    #             cv2.imwrite(output_file, im_output[k])
    # exit()

