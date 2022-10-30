
import torch
from torch import nn
import numpy as np

import torchvision


class Image_Detector(nn.Module):
    def __init__(self,model_name):
        super(Image_Detector,self).__init__()

        self.model = torchvision.models.efficientnet_b7(pretrained=False)
        self.model.features[0][0]=nn.Conv2d(36, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[1]=nn.Linear(3520,1)


        self.features=self.model.features
        self.avgpool=self.model.avgpool
        self.classifier=self.model.classifier
        
        self.model_2 = torchvision.models.mobilenet_v3_large(pretrained=False)
        self.model_2.features[0][0]=nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.features_2=self.model_2.features
        self.avgpool_2=self.model_2.avgpool
        
        self.sigmoid=nn.Sigmoid()

    def forward(self,x,xx):
        x=self.features(x)
        y=torch.squeeze(self.avgpool(x))
        
        xx=self.features_2(xx)
        xx=torch.squeeze(self.avgpool_2(xx))
        
        z=torch.cat([y,xx],dim=-1)
        
        z=self.classifier(z)
        
        z=self.sigmoid(z)
        
        return z
    




import os
from PIL import Image
# import tqdm
import shutil
import glob
import random
from random import Random
import cv2
from io import BytesIO
    
    
def process_image(src, cropsize_min = 150, cropsize_max = 2048, cropsize_ratio = (5,8), qf_range = (65, 100), output_size  = 200):


        img = Image.open(src).convert('RGB')
        height = img.size[1]
        width = img.size[0]

        # select the size of crop
        cropmax = min(min(width, height), cropsize_max)

        if cropmax<cropsize_min:
            print(src, width, height)
        assert cropmax>=cropsize_min

        cropmin = max(cropmax*cropsize_ratio[0]//cropsize_ratio[1], cropsize_min)
        cropsize = random.randint(cropmin, cropmax)

        # select the type of interpolation
        interp = Image.ANTIALIAS if cropsize>output_size else Image.CUBIC

        # select the position of the crop
        x1 = random.randint(0, width - cropsize)
        y1 = random.randint(0, height - cropsize)

        # select the jpeg quality factor
        qf = random.randint(*qf_range)

        # make cropping
        img = img.crop((x1, y1, x1+cropsize, y1+cropsize))
        assert img.size[0]==cropsize
        assert img.size[1]==cropsize

        # make resizing
        img = img.resize((output_size, output_size), interp)
        assert img.size[0]==output_size
        assert img.size[1]==output_size

        
        
        # make jpeg compression
        
        with BytesIO() as f:
            img.save(f, format='JPEG', quality = qf)
            f.seek(0)
            img_jpg = Image.open(f)
            img_jpg.load()
            
            return np.array(img_jpg)





def get_all_ifft(image_fft_r,image_fft_g,image_fft_b,total_mask,img_half):
    mask=np.ones((200,200))
    all_image=[]
    for k in range(total_mask):
        i=k*2
        mask[img_half-i-1:img_half+i+1,img_half-i-1:img_half+i+1]=0
        img_back_r = np.abs(np.fft.ifft2(image_fft_r*mask))
        img_back_g = np.abs(np.fft.ifft2(image_fft_g*mask))
        img_back_b = np.abs(np.fft.ifft2(image_fft_b*mask))
        
        all_image=all_image+[img_back_r.copy(),img_back_g.copy(),img_back_b.copy()]
        
    imgs=(np.stack(all_image,0)/255).astype(np.float32)
    imgs=torch.from_numpy(imgs)
    return imgs

def get_all_fft(image_fft_r,image_fft_g,image_fft_b):
    
    image_fft_rm=np.abs(image_fft_r)
    image_fft_gm=np.abs(image_fft_g)
    image_fft_bm=np.abs(image_fft_b)
    

    image_fft_rm=image_fft_rm/(np.max(image_fft_rm)+1e-10)
    image_fft_gm=image_fft_gm/(np.max(image_fft_gm)+1e-10)
    image_fft_bm=image_fft_bm/(np.max(image_fft_bm)+1e-10)
    
    image_fft_ra=np.angle(image_fft_r)/3.14
    image_fft_ga=np.angle(image_fft_g)/3.14
    image_fft_ba=np.angle(image_fft_b)/3.14
    
    all_image=[image_fft_rm,image_fft_gm,image_fft_bm,image_fft_ra,image_fft_ga,image_fft_ba]
    
    imgs=np.stack(all_image,0).astype(np.float32)
    
    imgs=torch.from_numpy(imgs)
    
    return imgs


from skimage.filters.rank import entropy
from skimage.morphology import disk

def get_all_entropy(image_gray,disks):
    image_entropy=[]
    for i in range(disks):
        image_e= entropy(image_gray, disk(i+1))
        image_entropy.append(image_e.copy())
        
    imgs=(np.stack(image_entropy,0)/5).astype(np.float32)
    imgs=torch.from_numpy(imgs)
    
    return imgs