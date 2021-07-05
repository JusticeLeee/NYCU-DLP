#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
input_size = 28   
hidden_size = 256 # or 512
latent_size = 32
conditional_size = 8
lr = 0.05
epochs = 500
kl_annealing_type='cycle'
time = 2
load = 0


# In[ ]:


import copy
from MyDataLoader import MyDataLoader
from torch.utils.data import DataLoader
from model import *
from utils import *
from train import *
if __name__=='__main__':
    
    #create dir
    if not os.path.exists("./results"):
        os.makedirs("./results")
        
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    #load data
    train_data = MyDataLoader(path='lab5_dataset/train.txt', train=True)
    train_data = DataLoader(train_data, batch_size =1, shuffle=True,num_workers=8)
    test_data  = MyDataLoader(path='lab5_dataset/test.txt', train=False)
    test_data = DataLoader(test_data, batch_size =1, shuffle=False,num_workers=8)
    tensor2string = MyDataLoader(path='lab5_dataset/test.txt', train=False).tensor2string
    MAX_LENGTH = 1000 # all words len < 1000
    CVAE=CVAE(input_size,hidden_size,latent_size,conditional_size,max_length=MAX_LENGTH).to(device)
    
    # load best model and evaluation bleu score & gaussian score  
    if(load==1):
        path = 'models/best_CVAE_32_ckpt'
        CVAE.load_state_dict(torch.load(path))
        
        conversion, BLEUscore = evaluate(CVAE, test_data, tensor2string)
        total_BLEUscore=0
        total_Gaussianscore=0
        test_time=5
        for i in range(test_time):
            conversion, BLEUscore = evaluate(CVAE, test_data, tensor2string)
            print('test.txt prediction:')
            for i in range(len(conversion)):
                print('input:',conversion[i][0])
                print('target:',conversion[i][1])
                print('prediction:',conversion[i][2])
                print()
            total_BLEUscore+=BLEUscore
        print(f'avg BLEUscore {total_BLEUscore/test_time:.2f}')
    
        for i in range(test_time):    
            # generate words
            generated_words=Gaussian_generate(CVAE,latent_size, tensor2string)
            Gaussianscore=Gaussian_score(generated_words)
            print('generate 100 words with 4 different tenses:')
            print(generated_words)
            total_Gaussianscore+=Gaussianscore
        print()
        print(f'avg Gaussianscore {total_Gaussianscore/test_time:.2f}')

        
    else:
        CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list,Gaussianscore_list=[],[],[],[],[],[]
        optimizer = optim.SGD(CVAE.parameters(),lr=lr)
        best_BLEUscore=0    
        for epoch in range(1,epochs+1):
            # get teacher_forcing rator & kl weight
            teacher_forcing_ratio = get_teacher_forcing_ratio(epoch, epochs)
            kl_weight = get_kl_weight(epoch, epochs, kl_annealing_type, time)

            #train & compute loss
            CEloss, KLloss,_ = train(CVAE, train_data, optimizer, teacher_forcing_ratio, kl_weight, tensor2string)
            CEloss_list.append(CEloss)
            KLloss_list.append(KLloss)
            teacher_forcing_ratio_list.append(teacher_forcing_ratio)
            kl_weight_list.append(kl_weight)
            print(f'epoch{epoch:>2d}/{epochs}  tf_ratio:{teacher_forcing_ratio:.2f}  kl_weight:{kl_weight:.2f}')
            print(f'CE:{CEloss:.4f} + KL:{KLloss:.4f} = {CEloss+KLloss:.4f}')

            """
            evalutation with test_data
            """
            predict,BLEUscore=evaluate(CVAE,test_data,tensor2string)
            #G
            generated_words=Gaussian_generate(CVAE,latent_size,tensor2string)
            Gaussianscore=Gaussian_score(generated_words)
            BLEUscore_list.append(BLEUscore)
            Gaussianscore_list.append(Gaussianscore)
            print(predict)
            print(f'BLEU socre:{BLEUscore:.4f} Gaussian score:{Gaussianscore:.4f}\n')

            """
            update best model
            """
            if( BLEUscore>=0.7 )and (Gaussianscore >=0.3):
                # save model
                torch.save(CVAE.state_dict(),os.path.join('models',f'best_epoch_{epoch}_BLEUscore_{BLEUscore}_Gaussianscore_{Gaussianscore}.ckpt'))
          
            """
            store results
            """
            fig=plot(epoch,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list)
            fig.savefig(os.path.join('results',f'result.png'))
            fig_withG=plot(epoch,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list,Gaussianscore_list)
            fig_withG.savefig(os.path.join('results',f'fig_with_gaussian.png'))


# In[ ]:




