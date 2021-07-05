import pandas as pd
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
    
        self.root = root
        # the type is numpy.ndarray
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        #get the path and load the img 
        path = self.root + self.img_name[index] + '.jpeg' 
#         img  = mpimg.imread(path)
        img  = Image.open(path).convert('RGB')

        #get label according to indx
        label = self.label[index]
        
        #set the trasnform
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Use transform make 
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        label = torch.from_numpy(np.array(label))

#         img = transforms.Normalize(img,(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         img = torchvision.transforms.ToPILImage(img)
#         img = torchvision.transforms.ToTensor()(img)

        return img, label

