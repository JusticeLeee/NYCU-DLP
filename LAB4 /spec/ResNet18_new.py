#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import torch
from torch.utils import data
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,plot_confusion_matrix


# %%


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


# %%


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
        # make the size 3*512*512 -> 1*3*512*512
        img = torch.unsqueeze(img, 0)
        # get the tensor form label
        label = torch.from_numpy(np.array(label))

#         img = transforms.Normalize(img,(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         img = torchvision.transforms.ToPILImage(img)
#         img = torchvision.transforms.ToTensor()(img)

        return img, label

train_data = RetinopathyLoader("data/", "train")
test_data  = RetinopathyLoader("data/", "test")
# print(train_data[0][0].shape)
# transform = transforms.Compose([transforms.ToPILImage()])
# transforms.ToPILImage()(train_data[0][0][0]).show()
# transform(train_data[0][0][0]).show()


# %%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# %%


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# redefine ResNet by spec
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#         self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # maxpool follow spec.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = nn.Linear(in_features=51200, out_features=5, bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # maxpool
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, kernel_size=7, stride=1, padding=0)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


# %%





# %%


def ResNet18_train(net,num_train=28099):
    true_ans = 0.0
    false_ans = 0.0
    confusion_yTrue = []
    confusion_yPred = []
    for i in range(num_train):
        inputs = train_data[i][0].to(device)
        labels = train_data[i][1].to(device)
        
        # make gradients be zero , default is accumulated 
        optimizer.zero_grad()
        #put the input to net , output size 1*1000 
        outputs = net(inputs)
        #make the labels to a tensor with size 1*1
        labels = torch.unsqueeze(labels, 0)
        #caclulate lose
        loss = criterion(outputs, labels)
#         print(loss.item())
        #caculate gradient
        loss.backward()
        #using gradient update weight
        optimizer.step()
        #weight decay
#         scheduler.step()
        net.eval()
        ground_true = labels.item()
        pred_y = torch.argmax(outputs).item()
#         print("pred= ",pred_y)
        confusion_yTrue.append(ground_true)
        confusion_yPred.append(pred_y)
        
        if ground_true == pred_y:
            true_ans = true_ans + 1
        else:
            false_ans = false_ans + 1
    return true_ans, false_ans, confusion_yTrue, confusion_yPred


# %%


def evalutation(net,data,num_test=7025):
    net.eval()
    true_ans = 0.0
    false_ans = 0.0
    confusion_yTrue = []
    confusion_yPred = []
    
    for i in range(num_test):
        inputs = train_data[i][0].to(device)
        labels = train_data[i][1].to(device)
        
        outputs = net(inputs)
        
        ground_true = labels.item()
        pred_y = torch.argmax(outputs).item()
#         print("pred= ",pred_y)

        confusion_yTrue.append(ground_true)
        confusion_yPred.append(pred_y)
        
        if ground_true == pred_y:
            true_ans = true_ans + 1
        else:
            false_ans = false_ans + 1
    return true_ans, false_ans, confusion_yTrue, confusion_yPred


# %%


# Training
def run(net,n=10):
#     if(net==pre_net):
#         print("pretrain network:")
#     else:
#         print("newtrain network:")
    train_epoch_list = []
    train_acc_list = []
    test_epoch_list = []
    test_acc_list = []
    f= open('ResNet18.new_txt','a')
    for epoch in range(10):
        print ("Train: epoch "+ str(epoch+1))
        start = time.time()

        #Traning
        train_epoch_list.append(epoch)
        result = ResNet18_train(net,len(train_data))
        # result = ResNet18_train(net,n)
        train_y_true = result[2]
        train_y_pred = result[3]
        ACC = (result[0]/(result[0]+result[1]))
        train_acc_list.append(ACC)
        f.write(str(ACC)+" ")
        print ("TrainAccuracy is : "+str(ACC))

        
        #Testing
        test_epoch_list.append(epoch)
        result = evalutation(net,test_data,len(test_data))
        # result = evalutation(net,test_data,n)
        test_y_true = result[2]
        test_y_pred = result[3]
        ACC = (result[0]/(result[0]+result[1]))
        test_acc_list.append(ACC)
        f.write(str(ACC)+"\n")
        print ("TestAccuracy is : "+str(ACC))
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
    return train_y_true,train_y_pred,test_y_true,test_y_pred,train_acc_list,test_acc_list


# %%


import numpy as np
def plot_confusion_matrix(y_true,y_pred,title):
    print(title+":")
    cm = confusion_matrix(result_old[0],result_old[1],labels=[0,1,2,3,4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1,2,3,4])
    disp.plot(cmap=plt.cm.Blues)


# %%


#new net
net = ResNet18().to(device)
criterion = F.cross_entropy
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=5e-4) # weight decay
result_new = run(net)
plot_confusion_matrix(result_new[0],result_new[1],title="new_train_confusion")
plot_confusion_matrix(result_new[2],result_new[3],title="new_test_confusion")
new_tran_accu = result_new[4]
new_test_accu = result_new[5]




