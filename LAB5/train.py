from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token=0
EOS_token=1

def compute_bleu(output, reference):
    """
    copy from sample code
    """
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def loss_function(predict_distribution,predict_output_length,target,mu,logvar):
    
    Criterion=nn.CrossEntropyLoss()
    CEloss=Criterion(predict_distribution[:predict_output_length],target[:predict_output_length])

    # KL(N(mu,variance)||N(0,1))
    KLloss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return CEloss, KLloss


def train(CVAE,train_data,optimizer,teacher_forcing_ratio,kl_weight,tensor2string):

    CVAE.train()
    total_CEloss=0
    total_KLloss=0
    total_BLEUscore=0
    
    for word_tensor,tense_tensor in train_data:
        optimizer.zero_grad()
        word_tensor,tense_tensor=word_tensor[0].to(device),tense_tensor[0].to(device)

        # init 
        h0 = CVAE.encoder.init_h0(CVAE.hidden_size - CVAE.conditional_size)
        c = CVAE.tense_embedding(tense_tensor).view(1, 1, -1)
        encoder_hidden_state = torch.cat((h0, c), dim=-1)
        encoder_cell_state = CVAE.encoder.init_c0()

        use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False
        
        #fowarding one word
        predict_output,predict_distribution,mean,logvar=CVAE(word_tensor,encoder_hidden_state,encoder_cell_state,c,use_teacher_forcing)
        CEloss,KLloss = loss_function(predict_distribution,predict_output.size(0),word_tensor.view(-1),mean,logvar)
        loss = CEloss + kl_weight * KLloss
        total_CEloss+=CEloss.item()
        total_KLloss+=KLloss.item()
        predict,target=tensor2string(predict_output),tensor2string(word_tensor)
        total_BLEUscore+=compute_bleu(predict,target)

        #update
        loss.backward()
        optimizer.step()

    return total_CEloss/len(train_data), total_KLloss/len(train_data), total_BLEUscore/len(train_data)


def evaluate(CVAE,test_data,tensor2string):

    CVAE.eval()
    words=[]
    total_BLEUscore=0
    with torch.no_grad():
        for in_word_tensor,in_tense_tensor,tar_word_tensor,tar_tense_tensor in test_data:
            in_word_tensor,in_tense_tensor=in_word_tensor[0].to(device),in_tense_tensor[0].to(device)
            tar_word_tensor,tar_tense_tensor=tar_word_tensor[0].to(device),tar_tense_tensor[0].to(device)

            # init hidden_state
            h0 = CVAE.encoder.init_h0(CVAE.hidden_size - CVAE.conditional_size)
            in_c = CVAE.tense_embedding(in_tense_tensor).view(1, 1, -1)
            encoder_hidden_state = torch.cat((h0, in_c), dim=-1)
            
            # init cell_state
            encoder_cell_state = CVAE.encoder.init_c0()

            # forwarding one word
            tar_c=CVAE.tense_embedding(tar_tense_tensor).view(1,1,-1)
            predict_output=CVAE.predict(in_word_tensor,encoder_hidden_state,encoder_cell_state,tar_c)
            target_word=tensor2string(tar_word_tensor)
            predict_word=tensor2string(predict_output)
            words.append([tensor2string(in_word_tensor),target_word,predict_word])
            total_BLEUscore+=compute_bleu(predict_word,target_word)

    return words, total_BLEUscore/len(test_data)

