import os
import numpy as np
import torch
import matplotlib.pyplot as plt
device=device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Gaussian_score(words):
    """
    copy from sample code
    """
    words_list = []
    score = 0
    yourpath = 'lab5_dataset/train.txt' #should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_teacher_forcing_ratio(epoch,epochs):
    # from 1.0 to 0.0
    teacher_forcing_ratio = 1.-(1./(epochs-1))*(epoch-1)
    return teacher_forcing_ratio

def get_kl_weight(epoch,epochs,kl_annealing_type,time):

    if kl_annealing_type == 'linear':
        return (1./(time-1))*(epoch-1)*0.1  if epoch<time else 0.1
    
    #cyclic
    else: 
        period = epochs//time
        epoch %= period
        KL_weight = sigmoid((epoch - period // 2) / (period // 10))* 0.1
        return KL_weight


def Gaussian_generate(CVAE,latent_size,tensor2string):
    CVAE.eval()
    words=[]
    
    with torch.no_grad():
        # 100 words with 4 tenses
        for i in range(100):
            latent = torch.randn(1, 1, latent_size).to(device)
            tmp = []
            for tense in range(4):
                word = tensor2string(CVAE.generate(latent, tense))
                tmp.append(word)

            words.append(tmp)
    return words

def plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list,Gaussainscore_list=None):
    
    x=range(1,epochs+1)
    target = [0.7 for i in range (epochs)]
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x,target,linestyle=':',label='BLEU_target_0.7')
    plt.plot(x,CEloss_list, label='CEloss')
    plt.plot(x,KLloss_list, label='KLloss')
    plt.plot(x,BLEUscore_list,label='BLEU score')

    plt.plot(x,teacher_forcing_ratio_list,linestyle=':',label='tf_ratio')
    plt.plot(x,kl_weight_list,linestyle=':',label='kl_weight')
    if(Gaussainscore_list!=None):
        plt.plot(x,Gaussainscore_list, label='Gaussainscore')
    plt.legend()

    return fig