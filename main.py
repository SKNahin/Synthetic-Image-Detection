import os
import pandas as pd
import numpy as np
import tqdm
import sys
from sys import argv


from matplotlib import pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2


from helper_1 import *


import torch
from torch import nn

from skimage.filters.rank import entropy
from skimage.morphology import disk



model = Image_Detector(model_name=None)
model.load_state_dict(torch.load('VIP_Final.pth'))

model = model.cuda()
model.eval()



input_csv = argv[1]  # input csv
output_csv = argv[2]  # output csv


root_db = os.path.dirname(input_csv) # directory of testset
tab = pd.read_csv(input_csv) # read input csv as table

for index, dat in tqdm.tqdm(tab.iterrows(), total=len(tab)): # for on images
    filename = os.path.join(root_db, dat['filename']) # filepath of an image
    
    image_rgb=cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    
    # image_rgb=process_image(filename)

    imgs=torch.from_numpy((image_rgb/255).astype(np.float32)).permute(2,0,1)

    image_gray_fr = np.fft.fft2(image_rgb[:,:,0])
    image_gray_fr = np.fft.fftshift(image_gray_fr)
    
    image_gray_fg = np.fft.fft2(image_rgb[:,:,1])
    image_gray_fg = np.fft.fftshift(image_gray_fg)
    
    image_gray_fb = np.fft.fft2(image_rgb[:,:,2])
    image_gray_fb = np.fft.fftshift(image_gray_fb)

    imgs_ifft=get_all_ifft(image_gray_fr,image_gray_fg,image_gray_fb,10,100)
    imgs_fft=get_all_fft(image_gray_fr,image_gray_fg,image_gray_fb)
    
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_entropy=get_all_entropy(image_gray,3)
    
    final_image=torch.cat([imgs,imgs_ifft,image_entropy],dim=0)


    final_image=torch.unsqueeze(final_image,0).cuda()
    imgs_fft=torch.unsqueeze(imgs_fft,0).cuda()
    
    
    with torch.no_grad():
        pred=model(final_image,imgs_fft)
    
    pred=torch.squeeze(pred).cpu().numpy()
    logit = (pred>0.5)*1 # TODO compute the logit for the image
    tab.loc[index,'logit'] = logit  # insert the logit in table

tab.to_csv(output_csv, index=False) # save the results as csv file
