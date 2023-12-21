import torch 
import numpy as np 
import torch.nn as nn
from PIL import Image
import os

some_list = os.listdir("data/Abi/frames")[:20]
# first_path =
image_paths = ["data/Abi/frames/" + file for file in some_list]

# image = Image.open(image_paths[0])
# print(image)
# print(np.asarray(image))
"""
NOTE:
    - THE IMAGES ARE OF SHAPE (1080, 1920), WITH 3 CHANNELS    
"""
# for img in image_paths:
    # print(img)
    # image = Image.open(img)
    # image = np.asarray(image)
    # print(image.shape)
model_path = "../weights/vitpose_base_coco_aic_mpii.pth"
model = torch.load(model_path)
if __name__ == "__main__":
    # print(some_list)
    # print(image_paths)
    print(model)
    # print(True)
