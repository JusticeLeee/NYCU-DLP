import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token=0
EOS_token=1

class CVAE(nn.Module):  # conditional VAE
    # Encoder
    class Encoder(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(CVAE.Encoder,self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size)

        def forward(self, input, hidden, cell):
            embedded = self.embedding(input).view(1,1,-1)  # view(1,1,-1) due to input of lstm must be (seq_len,batch,vec_dim)
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            return output, hidden, cell

        def init_h0(self,size):
            return torch.zeros(1, 1, size, device=device)

        def init_c0(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)

    # Decoder
    class Decoder(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(CVAE.Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size)
            self.hidden2input = nn.Linear(hidden_size, input_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden, cell):
            output = self.embedding(input).view(1, 1, -1)
            output = F.relu(output)
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
            output = self.softmax(self.hidden2input(output[0])) 
            return output, hidden, cell

        def init_h0(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)

        def init_c0(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)

    def __init__(self, input_size, hidden_size, latent_size, conditional_size,max_length=1000):
        """
        input_size: 28
        hidden_size: 256 or 512
        latent_size: 32
        conditional_size: 8
        """
        super(CVAE,self).__init__()
        self.encoder = self.Encoder(input_size, hidden_size)
        self.decoder = self.Decoder(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.conditional_size = conditional_size
        self.max_length=max_length
        
        self.tense_embedding = nn.Embedding(4, conditional_size)  # 4 tense
        # use linear layer make size match 
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        # latnet concate with condition
        self.latentcondition2hidden=nn.Linear(latent_size+conditional_size,hidden_size)

    def forward(self,input_tensor,encoder_hidden,encoder_cell,condition,use_teacher_forcing):
        
        """
        encoder 
        """
        input_length=input_tensor.size(0)
        for ei in range(input_length):
            # only pass 1 alphabet 
            _ ,encoder_hidden,encoder_cell=self.encoder(input_tensor[ei],encoder_hidden,encoder_cell)
            
        """
        middle sample
        """
        #calculate mean & logvar
        mean=self.hidden2mean(encoder_hidden)
        logvar=self.hidden2logvar(encoder_hidden)
        # use mean & logvar sample
        latent=self.reparameterization(mean,logvar) 
        decoder_hidden = self.latentcondition2hidden(torch.cat((latent, condition), dim=-1))
        decoder_cell = self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)    #the begin of decoder default as SOS_token


        """
        decoder 
        """
        predict_distribution=torch.zeros(input_length,self.input_size,device=device)
        predict_output = None
        for di in range(input_length):
            output,decoder_hidden,decoder_cell=self.decoder(decoder_input,decoder_hidden,decoder_cell)
            predict_distribution[di]=output[0]
            predict_class=output.topk(1)[1]
            predict_output=torch.cat((predict_output,predict_class)) if predict_output is not None else predict_class

            if use_teacher_forcing:  # use teacher forcing
                decoder_input=input_tensor[di]
            else:
                if predict_class.item() == EOS_token:
                    break
                decoder_input = predict_class

        return predict_output,predict_distribution,mean,logvar

    def predict(self,input_tensor,encoder_hidden,encoder_cell,c):

        """
        encoder 
        """
        input_length=input_tensor.size(0)
        for ei in range(input_length):
            _ ,encoder_hidden,encoder_cell=self.encoder(input_tensor[ei],encoder_hidden,encoder_cell)
            
        """
        middle sample
        """
        mean=self.hidden2mean(encoder_hidden)
        logvar=self.hidden2logvar(encoder_hidden)
        # sampling a point
        latent=self.reparameterization(mean,logvar)
        decoder_hidden = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell = self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)

        """
        decoder 
        """
        predict_output = None
        for di in range(self.max_length):
            output, decoder_hidden, decoder_cell = self.decoder(decoder_input,decoder_hidden,decoder_cell)
            predict_class = output.topk(1)[1]
            predict_output = torch.cat((predict_output, predict_class)) if predict_output is not None else predict_class

            if predict_class.item() == EOS_token:
                break
            decoder_input = predict_class

        return predict_output
    
    def generate(self,latent,tense):
        tense_tensor=torch.tensor([tense]).to(device)
        c=self.tense_embedding(tense_tensor).view(1, 1, -1)
        decoder_hidden = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell=self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)

        """
        decoder 
        """
        predict_output = None
        for di in range(self.max_length):
            output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden,
                                                                            decoder_cell)
            predict_class = output.topk(1)[1]
            predict_output = torch.cat((predict_output, predict_class)) if predict_output is not None else predict_class

            if predict_class.item() == EOS_token:
                break
            decoder_input = predict_class

        return predict_output

    def reparameterization(self,mean,logvar):
        """
        x ~ N(0,1)
        y = ax + b
        y ~ N(b,a^2) 
        """
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        latent=mean+eps*std
        return latent

