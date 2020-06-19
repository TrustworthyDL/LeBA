import torch
import os
import time
import sys
import pandas as pd
import numpy as np
import imp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import imagenet


def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Logger:
    def __init__(self, file, print=True):
        self.file = file
        local_time = time.strftime("%b%d_%H%M%S", time.localtime()) 
        self.file += local_time
        self.All_file = 'logs/All.log'
        
    def print(self, content='', end = '\n', file=None):
        if file is None:
            file = self.file
        with open(file, 'a') as f:
            if isinstance(content, str):
                f.write(content+end)
            else:
                old=sys.stdout 
                sys.stdout = f
                print(content)
                sys.stdout = old
        if file is None:
            self.print(content, file=self.All_file)
        print(content,end=end)

class ImageSet(Dataset):
    def __init__(self, df, input_dir, transformer):
        self.df = df
        self.transformer = transformer
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_name = self.df.iloc[item]['image_path']
        image_path = os.path.join(self.input_dir,image_name)
        image = torch.tensor(np.array(Image.open(image_path)).astype(np.float32).transpose((2,0,1)))/255.0
        label_idx = self.df.iloc[item]['label_idx']
        target_idx = self.df.iloc[item]['target_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label': label_idx+1,
            'target': target_idx+1,
            'filename': image_name
        }
        return sample

def load_images_data(input_dir,  batch_size=16, shuffle=False, label_file='labels2'):   #Only forward
    dev_data = pd.read_csv(input_dir+'/'+label_file,header=None, sep=' ',
                            names=['image_path','label_idx','target_idx'])
    transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                    std=[0.5, 0.5, 0.5]),
    ])
    datasets = ImageSet(dev_data, input_dir, transformer)
    dataloader =  DataLoader(datasets,
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=shuffle)
    return dataloader

def get_model(model_type):
    return imagenet.get_model(model_type)

def normalize(x, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    mean_t = torch.Tensor([0.5, 0.5, 0.5]).reshape([1,3,1,1]).to(x.device)
    std_t = torch.Tensor([0.5, 0.5, 0.5]).reshape([1,3,1,1]).to(x.device)
    y = (x-mean_t)/std_t
    return y


def get_preprocess(model):
    def preprocess(images):
        return normalize(images)
    def preprocess2(images):
        images = images*255.0
        VGG_MEAN = [123.68, 116.78, 103.94] 
        for i in range(3):
            images[:, i,:, :] = images[:, i,:, :] - VGG_MEAN[i]
        new_images = F.interpolate(images,224)
        return normalize(new_images)
    if model=='vgg16':
        return preprocess2
    else:
        return preprocess

