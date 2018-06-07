import cv2
from glob import glob
import numpy as np
from tqdm import tqdm

im = cv2.imread('/home/storage/lsy/fashion/FashionAI_Keypoint_Detection/wu_train/Images/blouse/ff210d1818f907693a03a6ea2eb39f77.jpg')

for fn in tqdm(glob('/home/storage/lsy/fashion/FashionAI_Keypoint_Detection/r1_train/Images/blouse/*.jpg')):
    im2 = cv2.imread(fn)
    if im.shape == im2.shape:
        if np.all(im==im2):
            print(fn)
