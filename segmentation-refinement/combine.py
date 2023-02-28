import cv2
import numpy as np
file0 = '/data/yliugu/output_check/output.png'
file1 = '/data/yliugu/output_check/output_1.png'
output_file = '/data/yliugu/output_check/combine.png'
image = cv2.imread(file0)
image_1 = cv2.imread(file1)


kernel = np.ones((7, 7), np.uint8)
image = cv2.erode(image.astype(float), kernel, iterations=1)
image_1 = cv2.dilate(image_1.astype(float), kernel, iterations=1)
output = (image + image_1) // 2
cv2.imwrite(output_file, output, )

